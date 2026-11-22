// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/scf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <unordered_map>
#include <stdexcept>

#include "dft/core/device.h"
#include "dft/scf/ions.h"
#include "dft/scf/potentials.h"
#include "dft/solver/davidson.h"

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

__global__ void add_kernel(std::size_t n, const double* a, const double* b, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// seconds between laps, the GPU finished
struct Clock {
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    double lap() {
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last).count();
        last = now;
        return seconds;
    }
};

// random bands decaying as 1 / (1 + |k+G|^2 / 2)
std::vector<Complex> random_bands(const std::vector<double>& kinetic, int bands, unsigned long long seed) {
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<Complex> psi(kinetic.size() * static_cast<std::size_t>(bands));
    for (int b = 0; b < bands; ++b)
        for (std::size_t g = 0; g < kinetic.size(); ++g) {
            const double scale = 1.0 / (1.0 + kinetic[g]);
            const double re = uniform(engine), im = uniform(engine);
            psi[static_cast<std::size_t>(b) * kinetic.size() + g] = make_cuDoubleComplex(scale * re, scale * im);
        }
    return psi;
}

// (up, down) components on the sphere to (total, magnetization) and back
void to_total(std::vector<Complex>& c, std::size_t n) {
    for (std::size_t g = 0; g < n; ++g) {
        const Complex up = c[g], down = c[n + g];
        c[g] = make_cuDoubleComplex(up.x + down.x, up.y + down.y);
        c[n + g] = make_cuDoubleComplex(up.x - down.x, up.y - down.y);
    }
}

void to_spins(std::vector<Complex>& c, std::size_t n) {
    for (std::size_t g = 0; g < n; ++g) {
        const Complex total = c[g], m = c[n + g];
        c[g] = make_cuDoubleComplex(0.5 * (total.x + m.x), 0.5 * (total.y + m.y));
        c[n + g] = make_cuDoubleComplex(0.5 * (total.x - m.x), 0.5 * (total.y - m.y));
    }
}

// the index of each Miller index of `to` in `from`, -1 where it is missing
std::vector<std::int64_t> match(const std::vector<Int3>& from, const std::vector<Int3>& to) {
    auto key = [](const Int3& m) {
        return (static_cast<std::int64_t>(m[0] + 4096) << 26) | (static_cast<std::int64_t>(m[1] + 4096) << 13) |
               static_cast<std::int64_t>(m[2] + 4096);
    };
    std::unordered_map<std::int64_t, std::int64_t> where;
    where.reserve(from.size());
    for (std::size_t i = 0; i < from.size(); ++i) where.emplace(key(from[i]), static_cast<std::int64_t>(i));
    std::vector<std::int64_t> out(to.size(), -1);
    for (std::size_t i = 0; i < to.size(); ++i) {
        const auto found = where.find(key(to[i]));
        if (found != where.end()) out[i] = found->second;
    }
    return out;
}

// column b of out (n, count) is column b of in (m, count) at the rows index,
// zero where the index is -1
__global__ void match_kernel(std::size_t n, std::size_t m, int count, const std::int64_t* index, const Complex* in,
                             Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t b = i / n, g = i % n;
    const std::int64_t from = index[g];
    out[i] = from < 0 ? make_cuDoubleComplex(0.0, 0.0) : in[b * m + static_cast<std::size_t>(from)];
}

bool fits(const ScfState* state, const ScfInput& input, int spins, int bands) {
    if (state == nullptr || !state->valid || state->atoms != input.atoms.size() || state->spins != spins ||
        state->bands != bands || state->kpoints.size() != input.kpoints.size())
        return false;
    for (std::size_t k = 0; k < input.kpoints.size(); ++k)
        for (std::size_t d = 0; d < 3; ++d)
            if (std::abs(state->kpoints[k][d] - input.kpoints[k][d]) > 1e-12) return false;
    return true;
}

}  // namespace

ScfResult run_scf(const ScfInput& input, const ScfOptions& options, ScfState* state) {
    const std::size_t points =
        static_cast<std::size_t>(input.grid[0]) * input.grid[1] * input.grid[2];
    const std::size_t nk = input.kpoints.size();
    const int nb = options.bands, ns = options.spins;
    if (ns != 1 && ns != 2) throw std::invalid_argument("one or two spins");
    const std::size_t spins = static_cast<std::size_t>(ns), bands = static_cast<std::size_t>(nb);
    if (nk == 0 || input.weights.size() != nk) throw std::invalid_argument("k-points and their weights");
    if (input.density.size() != spins * points) throw std::invalid_argument("rho_in of each spin on the grid");
    if (nb < 1) throw std::invalid_argument("at least one band");
    if (options.max_iterations < 1) throw std::invalid_argument("max_iterations must be positive");
    const double f_max = ns == 2 ? 1.0 : 2.0;
    const Lattice* basis = input.has_basis ? &input.basis : nullptr;
    Clock clock;
    Hamiltonian h(input.lattice, input.grid, input.cutoff, input.species, input.atoms, basis);
    Potentials pot(input.lattice, input.grid, input.density_cutoff, options.functional, ns, options.xc_threshold,
                   basis);
    // V_loc and rho_core of the atoms on the grid
    const std::size_t ng = pot.sphere().size();
    Ions ions(input.lattice, pot.sphere(), input.atoms, static_cast<int>(input.species.size()),
              input.local_factors, input.core_factors);
    DeviceArray<Complex> components(spins * ng);
    DeviceArray<double> local(points);
    ions.local_components(components.data());
    pot.from_sphere(components.data(), local.data(), 1);
    if (!input.local_slopes.empty()) ions.set_slopes(input.local_slopes, input.core_slopes);
    if (!input.atom_factors.empty()) ions.set_atom_factors(input.atom_factors);
    if (ions.has_core()) {
        DeviceArray<double> core(points);
        ions.core_components(components.data());
        pot.from_sphere(components.data(), core.data(), 1);
        std::vector<double> host(points);
        core.download(host.data(), points);
        pot.set_core(host.data());
    }
    Davidson davidson;
    // the bands of each spin and k-point, (s, k) at s nk + k, random at the
    // start and the same for both spins, so that a density without
    // magnetization keeps none
    std::vector<DeviceArray<Complex>> psi(spins * nk);
    const bool restart = fits(state, input, ns, nb);
    std::vector<std::vector<Int3>> millers(nk);
    for (std::size_t k = 0; k < nk; ++k) {
        h.set_k(input.kpoints[k]);
        if (bands > h.plane_waves()) throw std::invalid_argument("more bands than plane waves at a k-point");
        millers[k] = h.sphere().miller;
        if (restart) {
            // the bands of the last run on these plane waves
            const auto index = match(state->millers[k], millers[k]);
            DeviceArray<std::int64_t> where;
            where.upload(index.data(), index.size());
            const std::size_t n = h.plane_waves(), m = state->millers[k].size();
            for (std::size_t s = 0; s < spins; ++s) {
                psi[s * nk + k].resize(n * bands);
                match_kernel<<<blocks(n * bands), kThreads>>>(n, m, nb, where.data(), state->psi[s * nk + k].data(),
                                                              psi[s * nk + k].data());
                check_cuda(cudaGetLastError(), "the bands of the last run");
            }
            continue;
        }
        const auto start = random_bands(h.sphere().kinetic, nb, options.seed * 1000003ULL + k);
        for (std::size_t s = 0; s < spins; ++s) psi[s * nk + k].upload(start.data(), start.size());
    }
    // the mixer of (total, magnetization) on the sphere of the density
    std::vector<double> g2(ng);
    for (std::size_t g = 0; g < ng; ++g) g2[g] = 2.0 * pot.sphere().kinetic[g];
    Mixer mixer(g2, input.lattice.volume, ns, options.mixing);
    DeviceArray<double> rho_in, rho_out, hxc(spins * points), veff(points);
    rho_in.upload(input.density.data(), spins * points);
    rho_out.resize(spins * points);
    std::vector<Complex> in_g(spins * ng), out_g(spins * ng), last_in;
    pot.to_sphere(rho_in.data(), components.data(), ns);
    components.download(in_g.data(), in_g.size());
    if (ns == 2) to_total(in_g, ng);
    const std::vector<Complex> atoms_g = in_g;  // of the free atoms where they are
    if (restart) {
        // the last output density less its free atoms plus these
        const auto index = match(state->density_miller, pot.sphere().miller);
        const std::size_t m = state->density_miller.size();
        for (std::size_t c = 0; c < spins; ++c)
            for (std::size_t g = 0; g < ng; ++g) {
                const std::int64_t from = index[g];
                if (from < 0) continue;
                const Complex old = state->density[c * m + static_cast<std::size_t>(from)];
                const Complex free = state->atoms_density[c * m + static_cast<std::size_t>(from)];
                in_g[c * ng + g] = make_cuDoubleComplex(in_g[c * ng + g].x + old.x - free.x,
                                                        in_g[c * ng + g].y + old.y - free.y);
            }
        std::vector<Complex> start = in_g;
        if (ns == 2) to_spins(start, ng);
        components.upload(start.data(), start.size());
        pot.from_sphere(components.data(), rho_in.data(), ns);
    }
    ScfResult result;
    result.restarted = restart;
    result.ewald = input.ewald;
    // the k-points of each spin, their weights and eigenvalues
    std::vector<double> weights(spins * nk), eigenvalues(spins * nk * bands), values, w(bands);
    for (std::size_t s = 0; s < spins; ++s)
        std::copy(input.weights.begin(), input.weights.end(), weights.begin() + static_cast<std::ptrdiff_t>(s * nk));
    double tolerance = 1e-2, previous = std::numeric_limits<double>::infinity();
    int tight = nb;
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        // the potential of rho_in
        if (iteration == 1) result.times.setup += clock.lap();
        const PotentialEnergies e = pot.compute(rho_in.data(), hxc.data());
        result.times.potentials += clock.lap();
        last_in = in_g;
        // the bands
        ScfStep step;
        step.tolerance = tolerance;
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                add_kernel<<<blocks(points), kThreads>>>(points, local.data(), hxc.data() + s * points, veff.data());
                check_cuda(cudaGetLastError(), "the effective potential");
                h.set_potential(veff.data());
                const auto r = davidson.solve(h, psi[s * nk + k].data(), nb, values, tolerance, tight,
                                              std::max(tolerance, 1e-2), options.davidson_iterations,
                                              options.davidson_basis * nb);
                step.davidson += r.iterations;
                std::copy(values.begin(), values.end(),
                          eigenvalues.begin() + static_cast<std::ptrdiff_t>((s * nk + k) * bands));
            }
        }
        result.times.bands += clock.lap();
        const Occupation occ =
            occupy(eigenvalues, weights, nb, options.electrons, f_max, options.smearing, options.sigma);
        // the output density and the energy
        rho_out.zero();
        double band = 0.0;
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) {
                    w[b] = weights[s * nk + k] * f_max * occ.f[row + b];
                    band += w[b] * eigenvalues[row + b];
                }
                h.add_density(psi[s * nk + k].data(), nb, w.data(), rho_out.data() + s * points);
            }
        }
        const double energy = band - e.hxc_density + e.hartree + e.xc + input.ewald + occ.entropy;
        pot.to_sphere(rho_out.data(), components.data(), ns);
        components.download(out_g.data(), out_g.size());
        if (ns == 2) to_total(out_g, ng);
        result.times.density += clock.lap();
        const double accuracy = mixer.residual_energy(in_g, out_g);
        if (!std::isfinite(energy) || !std::isfinite(accuracy))
            throw std::runtime_error("the energy or the density is not finite");
        step.energy = energy;
        step.accuracy = accuracy;
        step.fermi = occ.fermi;
        step.magnetization = ns == 2 ? out_g[ng].x * input.lattice.volume : 0.0;
        result.steps.push_back(step);
        result.iterations = iteration;
        result.energy = energy;
        result.band = band;
        result.one_electron = band - e.hxc_density;
        result.hartree = e.hartree;
        result.xc = e.xc;
        result.entropy = occ.entropy;
        result.fermi = occ.fermi;
        result.lowest_empty = occ.lowest_empty;
        result.eigenvalues = eigenvalues;
        result.occupations = occ.f;
        const bool done = accuracy < options.density_tolerance && std::abs(energy - previous) < options.energy_tolerance;
        previous = energy;
        if (done) {
            result.converged = true;
            break;
        }
        // the next input density
        in_g = mixer.mix(in_g, out_g);
        result.times.mixing += clock.lap();
        std::vector<Complex> next = in_g;
        if (ns == 2) to_spins(next, ng);
        components.upload(next.data(), next.size());
        pot.from_sphere(components.data(), rho_in.data(), ns);
        // the bands that hold electrons, and the tolerance of the next step
        tight = 1;
        for (std::size_t row = 0; row < spins * nk; ++row)
            for (int b = nb - 1; b >= tight; --b)
                if (std::abs(occ.f[row * bands + static_cast<std::size_t>(b)]) > 1e-10) {
                    tight = b + 1;
                    break;
                }
        tolerance = std::clamp(0.01 * std::sqrt(accuracy), 1e-9, tolerance);
    }
    clock.lap();
    // the forces: the local term of rho_out, the core charge in V_xc of the
    // last rho_in (V_Hxc of each spin, averaged, less V_H), the projectors
    {
        DeviceArray<Complex> rho_g, vxc_g;
        rho_g.upload(out_g.data(), ng);
        pot.to_sphere(hxc.data(), components.data(), ns);
        std::vector<Complex> v(spins * ng);
        components.download(v.data(), v.size());
        for (std::size_t g = 0; g < ng; ++g) {
            Complex x = v[g];
            if (ns == 2) x = make_cuDoubleComplex(0.5 * (x.x + v[ng + g].x), 0.5 * (x.y + v[ng + g].y));
            const double g2_ = 2.0 * pot.sphere().kinetic[g];
            if (g2_ > 0.0) {
                const double f = 4.0 * 3.14159265358979323846 / g2_;
                x = make_cuDoubleComplex(x.x - f * last_in[g].x, x.y - f * last_in[g].y);
            }
            v[g] = x;
        }
        vxc_g.upload(v.data(), ng);
        result.forces = ions.forces(rho_g.data(), vxc_g.data());
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) w[b] = weights[s * nk + k] * f_max * result.occupations[row + b];
                h.nonlocal_forces(psi[s * nk + k].data(), nb, w.data(), result.forces);
            }
        }
    }
    result.times.forces += clock.lap();
    // V_Hxc of rho_out, for the correction of the forces and the stress (the
    // last potentials computed, whose gradients the stress of GGA takes)
    DeviceArray<double> hxc_out(spins * points);
    const PotentialEnergies e_out = pot.compute(rho_out.data(), hxc_out.data());
    if (ions.has_atom_factors()) {
        // dV = V_Hxc[rho_out] - V_Hxc[rho_in], averaged over the spins
        std::vector<double> a(spins * points), b(spins * points);
        hxc_out.download(a.data(), a.size());
        hxc.download(b.data(), b.size());
        std::vector<double> dv(points);
        for (std::size_t i = 0; i < points; ++i) {
            double x = 0.0;
            for (std::size_t s = 0; s < spins; ++s) x += a[s * points + i] - b[s * points + i];
            dv[i] = x / static_cast<double>(spins);
        }
        DeviceArray<double> dv_r;
        dv_r.upload(dv.data(), points);
        DeviceArray<Complex> dv_g(ng);
        pot.to_sphere(dv_r.data(), dv_g.data(), 1);
        const auto correction = ions.correction_forces(dv_g.data());
        for (std::size_t i = 0; i < correction.size(); ++i) result.forces[i] += correction[i];
    }
    result.times.forces += clock.lap();
    // the strain derivative of the energy of Kohn and Sham of the last bands
    if (ions.has_slopes()) {
        const auto& sphere = pot.sphere();
        const double omega = input.lattice.volume, pi = 3.14159265358979323846;
        const int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
        std::vector<double> gv(3 * ng);
        for (std::size_t g = 0; g < ng; ++g)
            for (int d = 0; d < 3; ++d) {
                double x = 0.0;
                for (int j = 0; j < 3; ++j)
                    x += sphere.miller[g][static_cast<std::size_t>(j)] *
                         input.lattice.b[static_cast<std::size_t>(j)][static_cast<std::size_t>(d)];
                gv[3 * g + static_cast<std::size_t>(d)] = x;
            }
        auto download = [&](void (Ions::*f)(Complex*) const) {
            (ions.*f)(components.data());
            std::vector<Complex> out(ng);
            components.download(out.data(), ng);
            return out;
        };
        std::array<double, 6> st{};
        auto add_diagonal = [&](double x) {
            for (int d = 0; d < 3; ++d) st[static_cast<std::size_t>(d)] += x;
        };
        // sum_G w(G) G_a G_b, G = 0 left out
        auto add_shells = [&](const std::function<double(std::size_t)>& weight) {
            for (std::size_t g = 1; g < ng; ++g) {
                const double w_ = weight(g);
                for (int d = 0; d < 6; ++d)
                    st[static_cast<std::size_t>(d)] += w_ * gv[3 * g + static_cast<std::size_t>(pairs[d][0])] *
                                                       gv[3 * g + static_cast<std::size_t>(pairs[d][1])];
            }
        };
        const std::vector<Complex>& rho = out_g;  // the total density first
        auto g2_of = [&](std::size_t g) { return 2.0 * sphere.kinetic[g]; };
        // Hartree: -delta E_H + 4 pi Omega sum |rho|^2 G_a G_b / G^4
        double hartree = 0.0;
        for (std::size_t g = 1; g < ng; ++g)
            hartree += (rho[g].x * rho[g].x + rho[g].y * rho[g].y) / g2_of(g);
        hartree *= 2.0 * pi * omega;
        add_diagonal(-hartree);
        add_shells([&](std::size_t g) {
            const double x = g2_of(g);
            return 4.0 * pi * omega * (rho[g].x * rho[g].x + rho[g].y * rho[g].y) / (x * x);
        });
        // local: -delta E_loc - Omega sum Re[conj(rho) S V'] G_a G_b / |G|
        const auto vloc = download(&Ions::local_components);
        const auto vslope = download(&Ions::local_slope_components);
        double eloc = 0.0;
        for (std::size_t g = 0; g < ng; ++g) eloc += rho[g].x * vloc[g].x + rho[g].y * vloc[g].y;
        add_diagonal(-omega * eloc);
        add_shells([&](std::size_t g) {
            return -omega * (rho[g].x * vslope[g].x + rho[g].y * vslope[g].y) / std::sqrt(g2_of(g));
        });
        // exchange and correlation of rho_out: delta [E_xc - int V_xc (rho + rho_core)]
        // - Omega sum Re[conj(V_xc) S c'] G_a G_b / |G| + the gradients of GGA
        const PotentialEnergies& e = e_out;
        pot.to_sphere(hxc_out.data(), components.data(), ns);
        std::vector<Complex> v(spins * ng);
        components.download(v.data(), v.size());
        for (std::size_t g = 0; g < ng; ++g) {
            Complex x = v[g];
            if (ns == 2) x = make_cuDoubleComplex(0.5 * (x.x + v[ng + g].x), 0.5 * (x.y + v[ng + g].y));
            if (g > 0) {
                const double f = 4.0 * pi / g2_of(g);
                x = make_cuDoubleComplex(x.x - f * rho[g].x, x.y - f * rho[g].y);
            }
            v[g] = x;
        }
        double vxc_core = 0.0;
        if (ions.has_core()) {
            const auto core = download(&Ions::core_components);
            const auto cslope = download(&Ions::core_slope_components);
            for (std::size_t g = 0; g < ng; ++g) vxc_core += v[g].x * core[g].x + v[g].y * core[g].y;
            vxc_core *= omega;
            add_shells([&](std::size_t g) {
                return -omega * (v[g].x * cslope[g].x + v[g].y * cslope[g].y) / std::sqrt(g2_of(g));
            });
        }
        add_diagonal(e.xc - (e.hxc_density - 2.0 * e.hartree) - vxc_core);
        const auto gga = pot.gga_strain();
        for (int d = 0; d < 6; ++d) st[static_cast<std::size_t>(d)] += gga[static_cast<std::size_t>(d)];
        // the kinetic energy and the projectors of the bands
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) w[b] = weights[s * nk + k] * f_max * result.occupations[row + b];
                h.strain_derivative(psi[s * nk + k].data(), nb, w.data(), st);
            }
        }
        result.has_strain = true;
        result.strain = st;
    }
    result.times.stress += clock.lap();
    if (state != nullptr) {
        state->valid = result.converged;
        state->atoms = input.atoms.size();
        state->spins = ns;
        state->bands = nb;
        state->kpoints = input.kpoints;
        state->millers = std::move(millers);
        state->psi = std::move(psi);
        state->density_miller = pot.sphere().miller;
        state->density = out_g;
        state->atoms_density = atoms_g;
    }
    result.density.resize(spins * points);
    rho_out.download(result.density.data(), result.density.size());
    if (ns == 2) {
        const double dv = input.lattice.volume / static_cast<double>(points);
        for (std::size_t i = 0; i < points; ++i) {
            const double m = result.density[i] - result.density[points + i];
            result.magnetization += m * dv;
            result.absolute_magnetization += std::abs(m) * dv;
        }
    }
    return result;
}

}  // namespace mdapy::dft

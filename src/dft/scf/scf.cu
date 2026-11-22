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
#include <memory>
#include <random>
#include <unordered_map>
#include <stdexcept>

#include "dft/core/device.h"
#include "dft/scf/augmentation.h"
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

// the width of the projector functions of a species
int width_of(const SpeciesProjectors& s) {
    int w = 0;
    for (const int l : s.angular) w += 2 * l + 1;
    return w;
}

// the strengths of a species as a block (width, width), D_pq between the
// functions of one l and m
std::vector<double> species_block(const SpeciesProjectors& s) {
    const int w = width_of(s);
    std::vector<double> out(static_cast<std::size_t>(w) * w, 0.0);
    const int count = static_cast<int>(s.angular.size());
    std::vector<int> first(static_cast<std::size_t>(count));
    for (int p = 0, f = 0; p < count; f += 2 * s.angular[static_cast<std::size_t>(p)] + 1, ++p)
        first[static_cast<std::size_t>(p)] = f;
    for (int p = 0; p < count; ++p)
        for (int q = 0; q < count; ++q) {
            const int l = s.angular[static_cast<std::size_t>(p)];
            if (l != s.angular[static_cast<std::size_t>(q)]) continue;
            for (int m = 0; m < 2 * l + 1; ++m)
                out[static_cast<std::size_t>(first[static_cast<std::size_t>(p)] + m) * w +
                    (first[static_cast<std::size_t>(q)] + m)] = s.dij[static_cast<std::size_t>(p * count + q)];
        }
    return out;
}

// the packed pairs a <= b of a block of width w added to it, symmetric
void add_packed(const double* packed, int w, double* block) {
    int k = 0;
    for (int a = 0; a < w; ++a)
        for (int b = a; b < w; ++b, ++k) {
            block[a * w + b] += packed[k];
            if (b != a) block[b * w + a] += packed[k];
        }
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
    // the bands on the smooth grid, the density and the potentials on the
    // dense one; the smooth sphere is the start of the dense one
    const Int3 smooth = input.smooth_grid[0] > 0 ? input.smooth_grid : input.grid;
    const bool two_grids = smooth != input.grid;
    const std::size_t points_s = static_cast<std::size_t>(smooth[0]) * smooth[1] * smooth[2];
    Hamiltonian h(input.lattice, smooth, input.cutoff, input.species, input.atoms, basis);
    Potentials pot(input.lattice, input.grid, input.density_cutoff, options.functional, ns, options.xc_threshold,
                   basis);
    std::unique_ptr<Potentials> pot_s;
    std::size_t ng_s = 0;
    if (two_grids) {
        pot_s = std::make_unique<Potentials>(input.lattice, smooth, 4.0 * input.cutoff, options.functional, ns,
                                             options.xc_threshold, basis);
        ng_s = pot_s->sphere().size();
        if (ng_s > pot.sphere().size()) throw std::invalid_argument("the dense grid is smaller than the smooth one");
        for (std::size_t g = 0; g < ng_s; ++g)
            if (pot_s->sphere().miller[g] != pot.sphere().miller[g])
                throw std::logic_error("the smooth sphere is not the start of the dense one");
    }
    DeviceArray<Complex> smooth_g(two_grids ? spins * ng_s : 0);
    DeviceArray<double> veff_s(two_grids ? points_s : 0), rho_s(two_grids ? spins * points_s : 0);
    DeviceArray<Complex> veff_dense_g(two_grids ? pot.sphere().size() : 0);
    // V on the dense grid to the Hamiltonian, through the smooth sphere with two
    auto set_potential = [&](const DeviceArray<double>& v) {
        if (!two_grids) {
            h.set_potential(v.data());
            return;
        }
        pot.to_sphere(v.data(), veff_dense_g.data(), 1);
        pot_s->from_sphere(veff_dense_g.data(), veff_s.data(), 1);
        h.set_potential(veff_s.data());
    };
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
    // ultrasoft species: their augmentation, the overlaps and the strengths of
    // the bare D for each atom
    std::vector<SpeciesAugmentation> augmented = input.augmentation;
    augmented.resize(input.species.size());
    Augmentation aug(input.lattice, pot.sphere(), input.atoms, augmented);
    const bool ultrasoft = aug.any();
    std::vector<double> bare;  // the blocks of D0 of the atoms
    std::vector<std::size_t> pair_offsets;
    if (ultrasoft) {
        std::vector<double> overlaps;
        std::size_t offset = 0;
        for (const Atom& atom : input.atoms) {
            const auto& sp = input.species[static_cast<std::size_t>(atom.species)];
            const int w = width_of(sp);
            const auto block = species_block(sp);
            bare.insert(bare.end(), block.begin(), block.end());
            std::vector<double> q(static_cast<std::size_t>(w) * w, 0.0);
            const auto& sa = augmented[static_cast<std::size_t>(atom.species)];
            if (sa.pairs() > 0) {
                if (sa.functions != w) throw std::invalid_argument("the augmentation and the projectors of a species disagree");
                const auto packed = aug.overlaps(atom.species);
                add_packed(packed.data(), w, q.data());
            }
            overlaps.insert(overlaps.end(), q.begin(), q.end());
            pair_offsets.push_back(offset);
            offset += static_cast<std::size_t>(w) * (w + 1) / 2;
        }
        pair_offsets.push_back(offset);
        h.set_overlaps(overlaps);
    }
    // the strengths of each spin: D0 + Omega sum_G conj(V_s(G)) Q(G) of a potential
    std::vector<std::vector<double>> strengths(spins);
    DeviceArray<Complex> veff_g(ng);
    DeviceArray<double> veff_r(ultrasoft ? points : 0);
    auto potential_of = [&](std::size_t s, const DeviceArray<double>& v) {
        add_kernel<<<blocks(points), kThreads>>>(points, local.data(), v.data() + s * points, veff_r.data());
        check_cuda(cudaGetLastError(), "the effective potential");
        pot.to_sphere(veff_r.data(), veff_g.data(), 1);
    };
    auto update_strengths = [&](const DeviceArray<double>& v) {
        if (!ultrasoft) return;
        for (std::size_t s = 0; s < spins; ++s) {
            potential_of(s, v);
            const auto packed = aug.strengths(veff_g.data());
            strengths[s] = bare;
            std::size_t at = 0;
            for (std::size_t i = 0; i < input.atoms.size(); ++i) {
                const int w = width_of(input.species[static_cast<std::size_t>(input.atoms[i].species)]);
                const std::size_t first = pair_offsets[i];
                if (pair_offsets[i + 1] > first && first < packed.size())
                    add_packed(packed.data() + first, w, strengths[s].data() + at);
                at += static_cast<std::size_t>(w) * w;
            }
        }
    };
    std::vector<std::vector<double>> becsum(spins);
    // PAW: the input occupations, mixed, and the one-center strengths of them,
    // of the callable given or of the GPU
    std::unique_ptr<OneCenter> centers;
    std::function<double(const std::vector<std::vector<double>>&, std::vector<std::vector<double>>&)> one =
        input.onecenter;
    if (!one && std::any_of(input.paw.begin(), input.paw.end(), [](const SpeciesOneCenter& x) { return x.mesh > 0; })) {
        std::vector<int> kinds, widths;
        for (const Atom& atom : input.atoms) {
            kinds.push_back(atom.species);
            widths.push_back(width_of(input.species[static_cast<std::size_t>(atom.species)]));
        }
        std::vector<SpeciesOneCenter> data = input.paw;
        data.resize(input.species.size());
        centers = std::make_unique<OneCenter>(std::move(data), kinds, widths, ns, options.functional,
                                              options.xc_threshold);
        one = [&centers](const std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& d) {
            return centers->compute(b, d);
        };
    }
    const bool paw = static_cast<bool>(one);
    if (paw && !ultrasoft) throw std::invalid_argument("PAW without augmentation");
    std::vector<std::vector<double>> becsum_in, onecenter(spins);
    double e1 = 0.0;
    auto add_onecenter = [&](const std::vector<std::vector<double>>& d) {
        for (std::size_t s = 0; s < spins; ++s) {
            if (d[s].size() != pair_offsets.back()) throw std::invalid_argument("the one-center strengths of each spin");
            std::size_t at = 0;
            for (std::size_t i = 0; i < input.atoms.size(); ++i) {
                const int w = width_of(input.species[static_cast<std::size_t>(input.atoms[i].species)]);
                add_packed(d[s].data() + pair_offsets[i], w, strengths[s].data() + at);
                at += static_cast<std::size_t>(w) * w;
            }
        }
    };
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
    if (paw) {
        becsum_in = restart && state->becsum.size() == spins ? state->becsum : input.becsum;
        if (becsum_in.size() != spins) throw std::invalid_argument("the first occupations of PAW of each spin");
        for (const auto& b : becsum_in)
            if (b.size() != pair_offsets.back()) throw std::invalid_argument("the first occupations of PAW of each atom");
    }
    ScfResult result;
    result.restarted = restart;
    result.ewald = input.ewald;
    // the k-points of each spin, their weights and eigenvalues
    std::vector<double> weights(spins * nk), eigenvalues(spins * nk * bands), values, w(bands);
    for (std::size_t s = 0; s < spins; ++s)
        std::copy(input.weights.begin(), input.weights.end(), weights.begin() + static_cast<std::ptrdiff_t>(s * nk));
    double tolerance = 1e-2, previous = std::numeric_limits<double>::infinity();
    // the occupations of the step before, which weight the tolerance of each
    // band: tolerance / sqrt(|f|), never above the loose one of empty bands
    std::vector<double> held;
    std::vector<double> limits(bands);
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        // the potential of rho_in
        if (iteration == 1) result.times.setup += clock.lap();
        const PotentialEnergies e = pot.compute(rho_in.data(), hxc.data());
        update_strengths(hxc);
        double paw_energy = 0.0;
        if (paw) {
            e1 = one(becsum_in, onecenter);
            add_onecenter(onecenter);
            paw_energy = e1;
            for (std::size_t s = 0; s < spins; ++s)
                for (std::size_t i = 0; i < becsum_in[s].size(); ++i) paw_energy -= onecenter[s][i] * becsum_in[s][i];
        }
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
                set_potential(veff);
                if (ultrasoft) h.set_strengths(strengths[s]);
                const double loose = std::max(tolerance, 1e-2);
                for (std::size_t b = 0; b < bands; ++b) {
                    const double f = held.empty() ? 1.0 : std::abs(held[(s * nk + k) * bands + b]);
                    limits[b] = f > 1e-10 ? std::min(loose, tolerance / std::sqrt(std::min(f, 1.0))) : loose;
                }
                const auto r = davidson.solve(h, psi[s * nk + k].data(), nb, values, limits,
                                              options.davidson_iterations, options.davidson_basis * nb);
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
        if (two_grids) rho_s.zero();
        if (ultrasoft)
            for (auto& b : becsum) b.assign(pair_offsets.back(), 0.0);
        double band = 0.0;
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) {
                    w[b] = weights[s * nk + k] * f_max * occ.f[row + b];
                    band += w[b] * eigenvalues[row + b];
                }
                h.add_density(psi[s * nk + k].data(), nb, w.data(),
                              two_grids ? rho_s.data() + s * points_s : rho_out.data() + s * points);
                if (ultrasoft) h.add_becsum(psi[s * nk + k].data(), nb, w.data(), becsum[s]);
            }
        }
        const double energy = band - e.hxc_density + e.hartree + e.xc + input.ewald + occ.entropy + paw_energy;
        if (two_grids) {
            // the smooth density on the start of the dense sphere
            pot_s->to_sphere(rho_s.data(), smooth_g.data(), ns);
            components.zero();
            for (std::size_t s = 0; s < spins; ++s)
                check_cuda(cudaMemcpy(components.data() + s * ng, smooth_g.data() + s * ng_s, ng_s * sizeof(Complex),
                                      cudaMemcpyDeviceToDevice),
                           "cudaMemcpy");
        } else {
            pot.to_sphere(rho_out.data(), components.data(), ns);
        }
        if (ultrasoft)  // the augmentation of each spin
            for (std::size_t s = 0; s < spins; ++s) aug.add_density(becsum[s], components.data() + s * ng);
        if (ultrasoft || two_grids) pot.from_sphere(components.data(), rho_out.data(), ns);
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
        // the next input density, and with PAW the occupations of (total,
        // magnetization) along
        if (paw) {
            const std::size_t n = pair_offsets.back();
            std::vector<double> x(spins * n), x_out(spins * n);
            for (std::size_t i = 0; i < n; ++i) {
                if (ns == 2) {
                    x[i] = becsum_in[0][i] + becsum_in[1][i];
                    x[n + i] = becsum_in[0][i] - becsum_in[1][i];
                    x_out[i] = becsum[0][i] + becsum[1][i];
                    x_out[n + i] = becsum[0][i] - becsum[1][i];
                } else {
                    x[i] = becsum_in[0][i];
                    x_out[i] = becsum[0][i];
                }
            }
            in_g = mixer.mix(in_g, out_g, &x, &x_out);
            for (std::size_t i = 0; i < n; ++i) {
                if (ns == 2) {
                    becsum_in[0][i] = 0.5 * (x[i] + x[n + i]);
                    becsum_in[1][i] = 0.5 * (x[i] - x[n + i]);
                } else {
                    becsum_in[0][i] = x[i];
                }
            }
        } else {
            in_g = mixer.mix(in_g, out_g);
        }
        result.times.mixing += clock.lap();
        std::vector<Complex> next = in_g;
        if (ns == 2) to_spins(next, ng);
        components.upload(next.data(), next.size());
        pot.from_sphere(components.data(), rho_in.data(), ns);
        // the occupations and the tolerance of the next step
        held = occ.f;
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
        if (ultrasoft)  // the augmentation in the potential of the bands, V of rho_in
            for (std::size_t s = 0; s < spins; ++s) {
                potential_of(s, hxc);
                const auto f = aug.forces(veff_g.data(), becsum[s]);
                for (std::size_t i = 0; i < f.size(); ++i) result.forces[i] += f[i];
            }
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) w[b] = weights[s * nk + k] * f_max * result.occupations[row + b];
                if (ultrasoft) h.set_strengths(strengths[s]);
                h.nonlocal_forces(psi[s * nk + k].data(), nb, w.data(), result.forces, eigenvalues.data() + row);
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
        if (ultrasoft) {
            // the strengths of the energy of Kohn and Sham of rho_out, and the
            // augmentation moving with the strain in V of rho_out
            update_strengths(hxc_out);
            if (paw) {
                // the one-center strengths of the output occupations
                std::vector<std::vector<double>> d(spins);
                one(becsum, d);
                add_onecenter(d);
            }
            for (std::size_t s = 0; s < spins; ++s) {
                potential_of(s, hxc_out);
                const auto a = aug.strain(veff_g.data(), becsum[s]);
                for (int d = 0; d < 6; ++d) st[static_cast<std::size_t>(d)] += a[static_cast<std::size_t>(d)];
            }
        }
        // the kinetic energy and the projectors of the bands
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) w[b] = weights[s * nk + k] * f_max * result.occupations[row + b];
                if (ultrasoft) h.set_strengths(strengths[s]);
                h.strain_derivative(psi[s * nk + k].data(), nb, w.data(), st, eigenvalues.data() + row);
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
        state->becsum = paw ? becsum : std::vector<std::vector<double>>{};
    }
    result.onecenter = e1;
    if (paw) result.becsum = becsum;
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

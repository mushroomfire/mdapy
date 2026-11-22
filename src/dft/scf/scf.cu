// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/scf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>

#include "dft/core/device.h"
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

}  // namespace

ScfResult run_scf(const ScfInput& input, const ScfOptions& options) {
    const std::size_t points =
        static_cast<std::size_t>(input.grid[0]) * input.grid[1] * input.grid[2];
    const std::size_t nk = input.kpoints.size();
    const int nb = options.bands, ns = options.spins;
    if (ns != 1 && ns != 2) throw std::invalid_argument("one or two spins");
    const std::size_t spins = static_cast<std::size_t>(ns), bands = static_cast<std::size_t>(nb);
    if (nk == 0 || input.weights.size() != nk) throw std::invalid_argument("k-points and their weights");
    if (input.local.size() != points || input.density.size() != spins * points ||
        (!input.core.empty() && input.core.size() != points))
        throw std::invalid_argument("V_loc, rho_core and rho_in of each spin on the grid");
    if (nb < 1) throw std::invalid_argument("at least one band");
    if (options.max_iterations < 1) throw std::invalid_argument("max_iterations must be positive");
    const double f_max = ns == 2 ? 1.0 : 2.0;
    Hamiltonian h(input.lattice, input.grid, input.cutoff, input.species, input.atoms);
    Potentials pot(input.lattice, input.grid, input.density_cutoff, options.functional, ns, options.xc_threshold);
    if (!input.core.empty()) pot.set_core(input.core.data());
    Davidson davidson;
    // the bands of each spin and k-point, (s, k) at s nk + k, random at the
    // start and the same for both spins, so that a density without
    // magnetization keeps none
    std::vector<DeviceArray<Complex>> psi(spins * nk);
    for (std::size_t k = 0; k < nk; ++k) {
        h.set_k(input.kpoints[k]);
        if (bands > h.plane_waves()) throw std::invalid_argument("more bands than plane waves at a k-point");
        const auto start = random_bands(h.sphere().kinetic, nb, options.seed * 1000003ULL + k);
        for (std::size_t s = 0; s < spins; ++s) psi[s * nk + k].upload(start.data(), start.size());
    }
    // the sphere of the density and its mixer, of (total, magnetization)
    const std::size_t ng = pot.sphere().size();
    std::vector<double> g2(ng);
    for (std::size_t g = 0; g < ng; ++g) g2[g] = 2.0 * pot.sphere().kinetic[g];
    Mixer mixer(g2, input.lattice.volume, ns, options.mixing);
    DeviceArray<double> local, rho_in, rho_out, hxc(spins * points), veff(points);
    local.upload(input.local.data(), points);
    rho_in.upload(input.density.data(), spins * points);
    rho_out.resize(spins * points);
    DeviceArray<Complex> components(spins * ng);
    std::vector<Complex> in_g(spins * ng), out_g(spins * ng);
    pot.to_sphere(rho_in.data(), components.data(), ns);
    components.download(in_g.data(), in_g.size());
    if (ns == 2) to_total(in_g, ng);
    ScfResult result;
    result.ewald = input.ewald;
    // the k-points of each spin, their weights and eigenvalues
    std::vector<double> weights(spins * nk), eigenvalues(spins * nk * bands), values, w(bands);
    for (std::size_t s = 0; s < spins; ++s)
        std::copy(input.weights.begin(), input.weights.end(), weights.begin() + static_cast<std::ptrdiff_t>(s * nk));
    double tolerance = 1e-2, previous = std::numeric_limits<double>::infinity();
    int tight = nb;
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        // the potential of rho_in
        const PotentialEnergies e = pot.compute(rho_in.data(), hxc.data());
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

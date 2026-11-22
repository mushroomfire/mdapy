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

}  // namespace

ScfResult run_scf(const ScfInput& input, const ScfOptions& options) {
    const std::size_t points =
        static_cast<std::size_t>(input.grid[0]) * input.grid[1] * input.grid[2];
    const std::size_t nk = input.kpoints.size();
    const int nb = options.bands;
    if (nk == 0 || input.weights.size() != nk) throw std::invalid_argument("k-points and their weights");
    if (input.local.size() != points || input.density.size() != points ||
        (!input.core.empty() && input.core.size() != points))
        throw std::invalid_argument("V_loc, rho_core and rho_in on the grid");
    if (nb < 1) throw std::invalid_argument("at least one band");
    if (options.max_iterations < 1) throw std::invalid_argument("max_iterations must be positive");
    const double f_max = 2.0;
    Hamiltonian h(input.lattice, input.grid, input.cutoff, input.species, input.atoms);
    Potentials pot(input.lattice, input.grid, input.density_cutoff, options.functional, 1, options.xc_threshold);
    if (!input.core.empty()) pot.set_core(input.core.data());
    Davidson davidson;
    // the bands of each k-point, random at the start
    std::vector<DeviceArray<Complex>> psi(nk);
    for (std::size_t k = 0; k < nk; ++k) {
        h.set_k(input.kpoints[k]);
        if (static_cast<std::size_t>(nb) > h.plane_waves())
            throw std::invalid_argument("more bands than plane waves at a k-point");
        const auto start = random_bands(h.sphere().kinetic, nb, options.seed * 1000003ULL + k);
        psi[k].upload(start.data(), start.size());
    }
    // the sphere of the density and its mixer
    const std::size_t ng = pot.sphere().size();
    std::vector<double> g2(ng);
    for (std::size_t g = 0; g < ng; ++g) g2[g] = 2.0 * pot.sphere().kinetic[g];
    Mixer mixer(g2, input.lattice.volume, 1, options.mixing);
    DeviceArray<double> local, rho_in, rho_out, hxc(points), veff(points);
    local.upload(input.local.data(), points);
    rho_in.upload(input.density.data(), points);
    rho_out.resize(points);
    DeviceArray<Complex> components(ng);
    std::vector<Complex> in_g(ng), out_g(ng);
    pot.to_sphere(rho_in.data(), components.data(), 1);
    components.download(in_g.data(), ng);
    ScfResult result;
    result.ewald = input.ewald;
    std::vector<double> eigenvalues(nk * static_cast<std::size_t>(nb)), values, weights(static_cast<std::size_t>(nb));
    double tolerance = 1e-2, previous = std::numeric_limits<double>::infinity();
    int tight = nb;
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        // the potential of rho_in
        const PotentialEnergies e = pot.compute(rho_in.data(), hxc.data());
        add_kernel<<<blocks(points), kThreads>>>(points, local.data(), hxc.data(), veff.data());
        check_cuda(cudaGetLastError(), "the effective potential");
        h.set_potential(veff.data());
        // the bands
        ScfStep step;
        step.tolerance = tolerance;
        for (std::size_t k = 0; k < nk; ++k) {
            h.set_k(input.kpoints[k]);
            const auto r = davidson.solve(h, psi[k].data(), nb, values, tolerance, tight,
                                          std::max(tolerance, 1e-2), options.davidson_iterations,
                                          options.davidson_basis * nb);
            step.davidson += r.iterations;
            std::copy(values.begin(), values.end(), eigenvalues.begin() + static_cast<std::ptrdiff_t>(k * nb));
        }
        const Occupation occ = occupy(eigenvalues, input.weights, nb, options.electrons, f_max, options.smearing,
                                      options.sigma);
        // the output density and the energy
        rho_out.zero();
        double band = 0.0;
        for (std::size_t k = 0; k < nk; ++k) {
            for (int b = 0; b < nb; ++b) {
                const double f = occ.f[k * nb + static_cast<std::size_t>(b)];
                weights[static_cast<std::size_t>(b)] = input.weights[k] * f_max * f;
                band += weights[static_cast<std::size_t>(b)] * eigenvalues[k * nb + static_cast<std::size_t>(b)];
            }
            h.set_k(input.kpoints[k]);
            h.add_density(psi[k].data(), nb, weights.data(), rho_out.data());
        }
        const double energy = band - e.hxc_density + e.hartree + e.xc + input.ewald + occ.entropy;
        pot.to_sphere(rho_out.data(), components.data(), 1);
        components.download(out_g.data(), ng);
        const double accuracy = mixer.residual_energy(in_g, out_g);
        if (!std::isfinite(energy) || !std::isfinite(accuracy))
            throw std::runtime_error("the energy or the density is not finite");
        step.energy = energy;
        step.accuracy = accuracy;
        step.fermi = occ.fermi;
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
        components.upload(in_g.data(), ng);
        pot.from_sphere(components.data(), rho_in.data(), 1);
        // the bands that hold electrons, and the tolerance of the next step
        tight = 1;
        for (std::size_t k = 0; k < nk; ++k)
            for (int b = nb - 1; b >= tight; --b)
                if (std::abs(occ.f[k * nb + static_cast<std::size_t>(b)]) > 1e-10) {
                    tight = b + 1;
                    break;
                }
        tolerance = std::clamp(0.01 * std::sqrt(accuracy), 1e-9, tolerance);
    }
    result.density.resize(points);
    rho_out.download(result.density.data(), points);
    return result;
}

}  // namespace mdapy::dft

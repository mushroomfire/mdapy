// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/eam.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mdapy::calculators {

Spline::Spline(const double* values, int n, double delta)
    : n_(n), inverse_delta_(1.0 / delta), coefficients_(static_cast<std::size_t>(n) + 1) {
    if (n < 5) {
        throw std::invalid_argument("a tabulated function needs at least 5 values");
    }
    auto& s = coefficients_;
    for (int m = 1; m <= n; ++m) {
        s[m][6] = values[m - 1];
    }
    s[1][5] = s[2][6] - s[1][6];
    s[2][5] = 0.5 * (s[3][6] - s[1][6]);
    s[n - 1][5] = 0.5 * (s[n][6] - s[n - 2][6]);
    s[n][5] = s[n][6] - s[n - 1][6];
    for (int m = 3; m <= n - 2; ++m) {
        s[m][5] = ((s[m - 2][6] - s[m + 2][6]) + 8.0 * (s[m + 1][6] - s[m - 1][6])) / 12.0;
    }
    for (int m = 1; m <= n - 1; ++m) {
        s[m][4] = 3.0 * (s[m + 1][6] - s[m][6]) - 2.0 * s[m][5] - s[m + 1][5];
        s[m][3] = s[m][5] + s[m + 1][5] - 2.0 * (s[m + 1][6] - s[m][6]);
    }
    s[n][4] = 0.0;
    s[n][3] = 0.0;
    for (int m = 1; m <= n; ++m) {
        s[m][2] = s[m][5] / delta;
        s[m][1] = 2.0 * s[m][4] / delta;
        s[m][0] = 3.0 * s[m][3] / delta;
    }
}

void Spline::evaluate(double x, double& value, double& derivative) const {
    double p = x * inverse_delta_ + 1.0;
    int m = static_cast<int>(p);
    m = std::max(1, std::min(m, n_ - 1));
    p -= m;
    p = std::min(p, 1.0);
    const auto& c = coefficients_[static_cast<std::size_t>(m)];
    value = ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
    derivative = (c[0] * p + c[1]) * p + c[2];
}

EamPotential make_eam_potential(const double* embedding, int elements, int nrho,
                                const double* density, int densities, const double* pair,
                                int pairs, int nr, double drho, double dr, double cutoff,
                                double rho_max) {
    EamPotential potential;
    // one density per element, or per element and element of the neighbour
    const bool per_pair = elements > 1 && densities == elements * elements;
    if ((densities != elements && !per_pair) || pairs != elements * (elements + 1) / 2) {
        throw std::invalid_argument("inconsistent EAM tables");
    }
    if (!(drho > 0.0) || !(dr > 0.0) || !(cutoff > 0.0)) {
        throw std::invalid_argument("EAM table spacings and cutoff must be positive");
    }
    potential.density_per_pair = per_pair;
    potential.elements = elements;
    potential.cutoff = cutoff;
    potential.rho_max = rho_max;
    for (int e = 0; e < elements; ++e) {
        potential.embedding.emplace_back(embedding + static_cast<std::size_t>(e) * nrho, nrho,
                                         drho);
    }
    for (int k = 0; k < densities; ++k) {
        potential.density.emplace_back(density + static_cast<std::size_t>(k) * nr, nr, dr);
    }
    for (int k = 0; k < pairs; ++k) {
        potential.pair.emplace_back(pair + static_cast<std::size_t>(k) * nr, nr, dr);
    }
    return potential;
}

double Spline::value(double x) const {
    double p = x * inverse_delta_ + 1.0;
    int m = static_cast<int>(p);
    m = std::max(1, std::min(m, n_ - 1));
    p -= m;
    p = std::min(p, 1.0);
    const auto& c = coefficients_[static_cast<std::size_t>(m)];
    return ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
}

void eam_densities(const EamPotential& potential, std::int64_t centers,
                   const std::int32_t* species, const double* positions, const double* matrix,
                   const std::int64_t* offsets, const index_t* neighbors,
                   const std::int8_t* shifts, double* energies, double* slopes,
                   const Control& control) {
    const double cutoff2 = potential.cutoff * potential.cutoff;
    parallel_for(
        centers, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                double sum = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double r[3];
                    const double r2 = neighbor::pair_vector(
                        positions + 3 * i, positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                        shifts + 3 * p, matrix, r);
                    if (r2 < cutoff2) {
                        sum += potential.density_of(species[neighbors[p]], species[i])
                                   .value(std::sqrt(r2));
                    }
                }
                double embedding;
                potential.embedding[species[i]].evaluate(sum, embedding, slopes[i]);
                if (sum > potential.rho_max) {
                    embedding += slopes[i] * (sum - potential.rho_max);
                }
                energies[i] = embedding;
            }
        },
        control);
}

namespace {

// The forces of the first centers atoms, with their virials when wanted.
template <bool kVirial>
void forces_of(const EamPotential& potential, std::int64_t centers, const std::int32_t* species,
               const double* positions, const double* matrix, const std::int64_t* offsets,
               const index_t* neighbors, const std::int8_t* shifts, const double* slope,
               double* energies, double* forces, double* virials, const Control& control,
               bool heat) {
    const double cutoff2 = potential.cutoff * potential.cutoff;
    parallel_for(
        centers, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                double f[3] = {0.0, 0.0, 0.0};
                double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                double pair_energy = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double v[3];
                    neighbor::pair_vector(positions + 3 * i,
                                          positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                                          shifts + 3 * p, matrix, v);
                    // del = x_i - x_j, as in LAMMPS
                    const double del[3] = {-v[0], -v[1], -v[2]};
                    const double r2 = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
                    if (!(r2 < cutoff2)) {
                        continue;
                    }
                    const index_t j = neighbors[p];
                    const int sj = species[j];
                    const double r = std::sqrt(r2);
                    // rho_ip: of the density i gives j; rho_jp: j gives i; one
                    // function for both between atoms of one element
                    double rho_i, rho_ip, rho_j, rho_jp, z2, z2p;
                    const Spline& gives = potential.density_of(si, sj);
                    const Spline& takes = potential.density_of(sj, si);
                    gives.evaluate(r, rho_i, rho_ip);
                    if (&takes == &gives) {
                        rho_jp = rho_ip;
                    } else {
                        takes.evaluate(r, rho_j, rho_jp);
                    }
                    potential.pair_of(si, sj).evaluate(r, z2, z2p);
                    const double recip = 1.0 / r;
                    const double phi = z2 * recip;
                    const double phip = z2p * recip - phi * recip;
                    const double psip = slope[i] * rho_jp + slope[j] * rho_ip + phip;
                    const double fpair = -psip * recip;
                    for (int a = 0; a < 3; ++a) {
                        f[a] += del[a] * fpair;
                    }
                    if constexpr (kVirial) {
                        // the virial: the pair force split in halves, as
                        // LAMMPS; or for the heat flux r_ij (x) dU_j/dr_i of
                        // Fan et al. (2015), U_j = F(rho_j) + phi / 2 over its
                        // pairs
                        const double share =
                            heat ? -(slope[j] * rho_ip + 0.5 * phip) * recip : 0.5 * fpair;
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                w[3 * a + b] += del[a] * del[b] * share;
                            }
                        }
                    }
                    pair_energy += 0.5 * phi;
                }
                energies[i] += pair_energy;
                for (int a = 0; a < 3; ++a) {
                    forces[3 * i + a] = f[a];
                }
                if constexpr (kVirial) {
                    for (int k = 0; k < 9; ++k) {
                        virials[9 * i + k] = w[k];
                    }
                }
            }
        },
        control);
}

}  // namespace

void eam_forces(const EamPotential& potential, std::int64_t centers, const std::int32_t* species,
                const double* positions, const double* matrix, const std::int64_t* offsets,
                const index_t* neighbors, const std::int8_t* shifts, const double* slopes,
                double* energies, double* forces, double* virials, const Control& control,
                bool heat) {
    if (virials != nullptr) {
        forces_of<true>(potential, centers, species, positions, matrix, offsets, neighbors,
                        shifts, slopes, energies, forces, virials, control, heat);
    } else {
        forces_of<false>(potential, centers, species, positions, matrix, offsets, neighbors,
                         shifts, slopes, energies, forces, virials, control, heat);
    }
}

void compute_eam(const EamPotential& potential, std::int64_t n, const std::int32_t* species,
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, const Control& control, bool heat) {
    std::vector<double> slopes(static_cast<std::size_t>(n), 0.0);  // F'(rho)
    // Electron density at each atom, from the density function of each
    // neighbor; then the forces, pair energies and virials, each atom
    // gathering from its own pairs.
    eam_densities(potential, n, species, positions, matrix, offsets, neighbors, shifts,
                  energies, slopes.data(), stage(control, 0, 2));
    eam_forces(potential, n, species, positions, matrix, offsets, neighbors, shifts,
               slopes.data(), energies, forces, virials, stage(control, 1, 2), heat);
}

}  // namespace mdapy::calculators

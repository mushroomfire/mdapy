// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/pair.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "neighbor/frame.h"

namespace mdapy::calculators {

namespace {

// The energy of a pair at r2 = r^2 below its cutoff, and fpair, the force on
// the first atom along del = x_i - x_j over r.
inline void lennard_jones(const double* c, double offset, double r2, double& energy,
                          double& fpair) {
    const double r2inv = 1.0 / r2;
    const double r6inv = r2inv * r2inv * r2inv;
    fpair = r6inv * (c[0] * r6inv - c[1]) * r2inv;
    energy = r6inv * (c[2] * r6inv - c[3]) - offset;
}

inline void morse(const double* c, double offset, double r2, double& energy, double& fpair) {
    const double r = std::sqrt(r2);
    const double dexp = std::exp(-c[1] * (r - c[2]));
    fpair = c[3] * (dexp * dexp - dexp) / r;
    energy = c[0] * (dexp * dexp - 2.0 * dexp) - offset;
}

inline void uhlenbeck_ford(const double* c, double offset, double r2, double& energy,
                           double& fpair) {
    const double expuf = std::exp(-r2 * c[1]);
    fpair = c[0] * expuf / (1.0 - expuf);
    energy = -c[2] * std::log(1.0 - expuf) - offset;
}

// The universal screening function of Ziegler, Biersack and Littmark: the
// weights of its four terms.
constexpr double kZblWeights[4] = {0.02817, 0.28022, 0.50986, 0.18175};

// The screened repulsion, switched off from the inner cutoff c[9] to the
// cutoff by a + b t added to its derivative and a t^3 / 3 + b t^4 / 4 to
// its energy, t = r - inner; the offset takes off its energy at the cutoff.
inline void zbl(const double* c, double offset, double r2, double& energy, double& fpair) {
    const double r = std::sqrt(r2);
    const double rinv = 1.0 / r;
    double sum = 0.0, slope = 0.0;
    for (int k = 0; k < 4; ++k) {
        const double term = kZblWeights[k] * std::exp(-c[k] * r);
        sum += term;
        slope -= c[k] * term;
    }
    double e = c[4] * sum * rinv;
    double de = c[4] * (slope - sum * rinv) * rinv;
    if (r2 > c[10]) {
        const double t = r - c[9];
        de += t * t * (c[5] + c[6] * t);
        e += t * t * t * (c[7] + c[8] * t);
    }
    fpair = -de * rinv;
    energy = e - offset;
}

template <PairStyle Style>
inline void pair_term(const double* c, double offset, double r2, double& e, double& fpair) {
    if constexpr (Style == PairStyle::LennardJones) {
        lennard_jones(c, offset, r2, e, fpair);
    } else if constexpr (Style == PairStyle::Morse) {
        morse(c, offset, r2, e, fpair);
    } else if constexpr (Style == PairStyle::UhlenbeckFord) {
        uhlenbeck_ford(c, offset, r2, e, fpair);
    } else {
        zbl(c, offset, r2, e, fpair);
    }
}

template <PairStyle Style, bool kVirial>
void compute(const PairPotential& potential, std::int64_t n, const std::int32_t* species,
             const double* positions, const double* matrix, const std::int64_t* offsets,
             const index_t* neighbors, const std::int8_t* shifts, double* energies,
             double* forces, double* virials, const Control& control) {
    const auto t = static_cast<std::int64_t>(potential.species);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t si = species[i];
                double f[3] = {0.0, 0.0, 0.0};
                double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                double energy = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double v[3];
                    neighbor::pair_vector(positions + 3 * i,
                                          positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                                          shifts + 3 * p, matrix, v);
                    const double del[3] = {-v[0], -v[1], -v[2]};  // x_i - x_j, as in LAMMPS
                    const double r2 = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
                    const std::int64_t k = si * t + species[neighbors[p]];
                    if (!(r2 < potential.cutsq[static_cast<std::size_t>(k)])) {
                        continue;
                    }
                    const double* c =
                        potential.coefficients.data() + pair_width(Style) * k;
                    const double offset = potential.offset[static_cast<std::size_t>(k)];
                    double e, fpair;
                    pair_term<Style>(c, offset, r2, e, fpair);
                    for (int a = 0; a < 3; ++a) {
                        f[a] += del[a] * fpair;
                        if constexpr (kVirial) {
                            for (int b = 0; b < 3; ++b) {
                                w[3 * a + b] += 0.5 * del[a] * del[b] * fpair;
                            }
                        }
                    }
                    energy += 0.5 * e;
                }
                energies[i] = energy;
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

template <PairStyle Style>
void compute_style(const PairPotential& potential, std::int64_t n, const std::int32_t* species,
                   const double* positions, const double* matrix, const std::int64_t* offsets,
                   const index_t* neighbors, const std::int8_t* shifts, double* energies,
                   double* forces, double* virials, const Control& control) {
    if (virials != nullptr) {
        compute<Style, true>(potential, n, species, positions, matrix, offsets, neighbors,
                             shifts, energies, forces, virials, control);
    } else {
        compute<Style, false>(potential, n, species, positions, matrix, offsets, neighbors,
                              shifts, energies, forces, virials, control);
    }
}

// On one thread, each pair once among the processes of MPI
// (calculators/once.h): the forces, energies and virials that the pairs give
// every atom, n of them, ghosts among them.
template <PairStyle Style>
void compute_once(const PairPotential& potential, std::int64_t n, std::int64_t centers,
                  const std::int32_t* species, const double* positions, const double* matrix,
                  const std::int64_t* offsets, const index_t* neighbors,
                  const std::int8_t* shifts, double* energies,
                  double* forces, double* virials) {
    const auto t = static_cast<std::int64_t>(potential.species);
    std::fill(energies, energies + n, 0.0);
    std::fill(forces, forces + 3 * n, 0.0);
    if (virials != nullptr) {
        std::fill(virials, virials + 9 * n, 0.0);
    }
    for (std::int64_t i = 0; i < centers; ++i) {
        const std::int64_t si = species[i];
        for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
            const auto j = static_cast<std::int64_t>(neighbors[p]);
            double v[3];
            neighbor::pair_vector(positions + 3 * i, positions + 3 * j, shifts + 3 * p, matrix, v);
            const double del[3] = {-v[0], -v[1], -v[2]};  // x_i - x_j
            const double r2 = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
            const std::int64_t k = si * t + species[j];
            if (!(r2 < potential.cutsq[static_cast<std::size_t>(k)])) {
                continue;
            }
            const double* c = potential.coefficients.data() + pair_width(Style) * k;
            const double offset = potential.offset[static_cast<std::size_t>(k)];
            double e, fpair;
            pair_term<Style>(c, offset, r2, e, fpair);
            for (int a = 0; a < 3; ++a) {
                forces[3 * i + a] += del[a] * fpair;
                forces[3 * j + a] -= del[a] * fpair;
            }
            energies[i] += 0.5 * e;
            energies[j] += 0.5 * e;
            if (virials != nullptr) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        const double w = 0.5 * del[a] * del[b] * fpair;
                        virials[9 * i + 3 * a + b] += w;
                        virials[9 * j + 3 * a + b] += w;
                    }
                }
            }
        }
    }
}

}  // namespace

PairPotential make_pair_potential(PairStyle style, int species, const double* cutsq,
                                  const double* offset, const double* coefficients) {
    PairPotential potential;
    potential.style = style;
    potential.species = species;
    const auto t = static_cast<std::size_t>(species);
    potential.cutsq.assign(cutsq, cutsq + t * t);
    potential.offset.assign(offset, offset + t * t);
    potential.coefficients.assign(
        coefficients, coefficients + static_cast<std::size_t>(pair_width(style)) * t * t);
    for (const double c2 : potential.cutsq) {
        if (!(c2 >= 0.0)) {
            throw std::invalid_argument("cutoffs must not be negative");
        }
        potential.cutoff = std::max(potential.cutoff, std::sqrt(c2));
    }
    return potential;
}

void compute_pair(const PairPotential& potential, std::int64_t n, const std::int32_t* species,
                  const double* positions, const double* matrix, const std::int64_t* offsets,
                  const index_t* neighbors, const std::int8_t* shifts, double* energies,
                  double* forces, double* virials, const Control& control) {
    if (potential.style == PairStyle::LennardJones) {
        compute_style<PairStyle::LennardJones>(potential, n, species, positions, matrix, offsets,
                                         neighbors, shifts, energies, forces, virials, control);
    } else if (potential.style == PairStyle::Morse) {
        compute_style<PairStyle::Morse>(potential, n, species, positions, matrix, offsets, neighbors,
                                  shifts, energies, forces, virials, control);
    } else if (potential.style == PairStyle::UhlenbeckFord) {
        compute_style<PairStyle::UhlenbeckFord>(potential, n, species, positions, matrix, offsets,
                                          neighbors, shifts, energies, forces, virials,
                                          control);
    } else {
        compute_style<PairStyle::ZBL>(potential, n, species, positions, matrix, offsets,
                                      neighbors, shifts, energies, forces, virials, control);
    }
}

void compute_pair_once(const PairPotential& potential, std::int64_t n, std::int64_t centers,
                       const std::int32_t* species, const double* positions,
                       const double* matrix, const std::int64_t* offsets,
                       const index_t* neighbors, const std::int8_t* shifts,
                       double* energies, double* forces,
                       double* virials) {
    if (potential.style == PairStyle::LennardJones) {
        compute_once<PairStyle::LennardJones>(potential, n, centers, species, positions, matrix,
                                              offsets, neighbors, shifts, energies, forces,
                                              virials);
    } else if (potential.style == PairStyle::Morse) {
        compute_once<PairStyle::Morse>(potential, n, centers, species, positions, matrix,
                                       offsets, neighbors, shifts, energies, forces,
                                       virials);
    } else if (potential.style == PairStyle::UhlenbeckFord) {
        compute_once<PairStyle::UhlenbeckFord>(potential, n, centers, species, positions,
                                               matrix, offsets, neighbors, shifts,
                                               energies, forces, virials);
    } else {
        compute_once<PairStyle::ZBL>(potential, n, centers, species, positions, matrix,
                                     offsets, neighbors, shifts, energies, forces, virials);
    }
}

}  // namespace mdapy::calculators

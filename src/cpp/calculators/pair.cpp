// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/pair.h"

#include <cmath>

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

template <PairStyle Style>
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
                    const double* c = potential.coefficients.data() + 4 * k;
                    const double offset = potential.offset[static_cast<std::size_t>(k)];
                    double e, fpair;
                    if constexpr (Style == PairStyle::LennardJones) {
                        lennard_jones(c, offset, r2, e, fpair);
                    } else {
                        morse(c, offset, r2, e, fpair);
                    }
                    for (int a = 0; a < 3; ++a) {
                        f[a] += del[a] * fpair;
                        for (int b = 0; b < 3; ++b) {
                            w[3 * a + b] += 0.5 * del[a] * del[b] * fpair;
                        }
                    }
                    energy += 0.5 * e;
                }
                energies[i] = energy;
                for (int a = 0; a < 3; ++a) {
                    forces[3 * i + a] = f[a];
                }
                for (int k = 0; k < 9; ++k) {
                    virials[9 * i + k] = w[k];
                }
            }
        },
        control);
}

}  // namespace

void compute_pair(const PairPotential& potential, std::int64_t n, const std::int32_t* species,
                  const double* positions, const double* matrix, const std::int64_t* offsets,
                  const index_t* neighbors, const std::int8_t* shifts, double* energies,
                  double* forces, double* virials, const Control& control) {
    if (potential.style == PairStyle::LennardJones) {
        compute<PairStyle::LennardJones>(potential, n, species, positions, matrix, offsets,
                                         neighbors, shifts, energies, forces, virials, control);
    } else {
        compute<PairStyle::Morse>(potential, n, species, positions, matrix, offsets, neighbors,
                                  shifts, energies, forces, virials, control);
    }
}

}  // namespace mdapy::calculators

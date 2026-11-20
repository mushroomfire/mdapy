// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/adp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "neighbor/frame.h"

namespace mdapy::calculators {

AdpPotential make_adp_potential(const double* embedding, int elements, int nrho,
                                const double* density, const double* pair, const double* dipole,
                                const double* quadrupole, int nr, double drho, double dr,
                                double cutoff) {
    if (!(drho > 0.0) || !(dr > 0.0) || !(cutoff > 0.0)) {
        throw std::invalid_argument("ADP table spacings and cutoff must be positive");
    }
    AdpPotential potential;
    potential.elements = elements;
    potential.cutoff = cutoff;
    const auto row = static_cast<std::size_t>(nr);
    for (int e = 0; e < elements; ++e) {
        potential.embedding.emplace_back(embedding + static_cast<std::size_t>(e) * nrho, nrho,
                                         drho);
        potential.density.emplace_back(density + e * row, nr, dr);
    }
    const auto pairs = static_cast<std::size_t>(elements * (elements + 1) / 2);
    for (std::size_t k = 0; k < pairs; ++k) {
        potential.pair.emplace_back(pair + k * row, nr, dr);
        potential.dipole.emplace_back(dipole + k * row, nr, dr);
        potential.quadrupole.emplace_back(quadrupole + k * row, nr, dr);
    }
    return potential;
}

void adp_densities(const AdpPotential& potential, std::int64_t centers,
                   const std::int32_t* species, const double* positions, const double* matrix,
                   const std::int64_t* offsets, const index_t* neighbors,
                   const std::int8_t* shifts, double* energies, double* state,
                   const Control& control) {
    const double cutoff2 = potential.cutoff * potential.cutoff;
    std::fill(state, state + kState * static_cast<std::size_t>(centers), 0.0);
    // The density, the dipole and the quadrupole at each atom, and its energy
    parallel_for(
        centers, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                double rho = 0.0;
                double* m = &state[kState * static_cast<std::size_t>(i) + 1];
                double* l = &state[kState * static_cast<std::size_t>(i) + 4];
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double v[3];
                    const double r2 = neighbor::pair_vector(
                        positions + 3 * i, positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                        shifts + 3 * p, matrix, v);
                    if (!(r2 < cutoff2)) {
                        continue;
                    }
                    const double del[3] = {-v[0], -v[1], -v[2]};  // x_i - x_j
                    const int sj = species[neighbors[p]];
                    const double r = std::sqrt(r2);
                    double value, derivative, u, w;
                    potential.density[static_cast<std::size_t>(sj)].evaluate(r, value, derivative);
                    rho += value;
                    const std::size_t k = potential.pair_index(si, sj);
                    potential.dipole[k].evaluate(r, u, derivative);
                    potential.quadrupole[k].evaluate(r, w, derivative);
                    for (int a = 0; a < 3; ++a) {
                        m[a] += u * del[a];
                    }
                    l[0] += w * del[0] * del[0];
                    l[1] += w * del[1] * del[1];
                    l[2] += w * del[2] * del[2];
                    l[3] += w * del[1] * del[2];
                    l[4] += w * del[0] * del[2];
                    l[5] += w * del[0] * del[1];
                }
                double embedding;
                potential.embedding[static_cast<std::size_t>(si)].evaluate(
                    rho, embedding, state[kState * static_cast<std::size_t>(i)]);
                const double nu = l[0] + l[1] + l[2];
                energies[i] = embedding + 0.5 * (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) +
                              0.5 * (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]) +
                              1.0 * (l[3] * l[3] + l[4] * l[4] + l[5] * l[5]) -
                              1.0 / 6.0 * nu * nu;
            }
        },
        control);

}

namespace {

template <bool kVirial>
void adp_forces_of(const AdpPotential& potential, std::int64_t centers,
                   const std::int32_t* species, const double* positions, const double* matrix,
                   const std::int64_t* offsets, const index_t* neighbors,
                   const std::int8_t* shifts, const double* state, double* energies,
                   double* forces, double* virials, const Control& control) {
    const double cutoff2 = potential.cutoff * potential.cutoff;
    // Forces, pair energies and virials; each atom gathers from its own pairs.
    parallel_for(
        centers, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                double f[3] = {0.0, 0.0, 0.0};
                double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                double pair_energy = 0.0;
                const double* mi = &state[kState * static_cast<std::size_t>(i) + 1];
                const double* li = &state[kState * static_cast<std::size_t>(i) + 4];
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double v[3];
                    const double r2 = neighbor::pair_vector(
                        positions + 3 * i, positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                        shifts + 3 * p, matrix, v);
                    if (!(r2 < cutoff2)) {
                        continue;
                    }
                    const double delx = -v[0], dely = -v[1], delz = -v[2];
                    const index_t j = neighbors[p];
                    const int sj = species[j];
                    const double r = std::sqrt(r2);
                    double rho_i, rhoip, rho_j, rhojp, z2, z2p, u2, u2p, w2, w2p;
                    potential.density[static_cast<std::size_t>(si)].evaluate(r, rho_i, rhoip);
                    potential.density[static_cast<std::size_t>(sj)].evaluate(r, rho_j, rhojp);
                    const std::size_t k = potential.pair_index(si, sj);
                    potential.pair[k].evaluate(r, z2, z2p);
                    potential.dipole[k].evaluate(r, u2, u2p);
                    potential.quadrupole[k].evaluate(r, w2, w2p);
                    const double recip = 1.0 / r;
                    const double phi = z2 * recip;
                    const double phip = z2p * recip - phi * recip;
                    const double psip = state[kState * static_cast<std::size_t>(i)] * rhojp +
                                        state[kState * static_cast<std::size_t>(j)] * rhoip + phip;
                    const double fpair = -psip * recip;
                    const double* mj = &state[kState * static_cast<std::size_t>(j) + 1];
                    const double* lj = &state[kState * static_cast<std::size_t>(j) + 4];
                    const double delmux = mi[0] - mj[0], delmuy = mi[1] - mj[1],
                                 delmuz = mi[2] - mj[2];
                    const double trdelmu = delmux * delx + delmuy * dely + delmuz * delz;
                    const double sxx = li[0] + lj[0], syy = li[1] + lj[1], szz = li[2] + lj[2];
                    const double syz = li[3] + lj[3], sxz = li[4] + lj[4], sxy = li[5] + lj[5];
                    const double tradellam = sxx * delx * delx + syy * dely * dely +
                                             szz * delz * delz + 2.0 * sxy * delx * dely +
                                             2.0 * sxz * delx * delz + 2.0 * syz * dely * delz;
                    const double nu = sxx + syy + szz;
                    const double adpx = delmux * u2 + trdelmu * u2p * delx * recip +
                                        2.0 * w2 * (sxx * delx + sxy * dely + sxz * delz) +
                                        w2p * delx * recip * tradellam -
                                        1.0 / 3.0 * nu * (w2p * r + 2.0 * w2) * delx;
                    const double adpy = delmuy * u2 + trdelmu * u2p * dely * recip +
                                        2.0 * w2 * (sxy * delx + syy * dely + syz * delz) +
                                        w2p * dely * recip * tradellam -
                                        1.0 / 3.0 * nu * (w2p * r + 2.0 * w2) * dely;
                    const double adpz = delmuz * u2 + trdelmu * u2p * delz * recip +
                                        2.0 * w2 * (sxz * delx + syz * dely + szz * delz) +
                                        w2p * delz * recip * tradellam -
                                        1.0 / 3.0 * nu * (w2p * r + 2.0 * w2) * delz;
                    const double force[3] = {delx * fpair - adpx, dely * fpair - adpy,
                                             delz * fpair - adpz};
                    const double del[3] = {delx, dely, delz};
                    for (int a = 0; a < 3; ++a) {
                        f[a] += force[a];
                        if constexpr (kVirial) {
                            for (int b = 0; b < 3; ++b) {
                                w[3 * a + b] += 0.5 * del[a] * force[b];
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

void adp_forces(const AdpPotential& potential, std::int64_t centers, const std::int32_t* species,
                const double* positions, const double* matrix, const std::int64_t* offsets,
                const index_t* neighbors, const std::int8_t* shifts, const double* state,
                double* energies, double* forces, double* virials, const Control& control) {
    if (virials != nullptr) {
        adp_forces_of<true>(potential, centers, species, positions, matrix, offsets, neighbors,
                            shifts, state, energies, forces, virials, control);
    } else {
        adp_forces_of<false>(potential, centers, species, positions, matrix, offsets, neighbors,
                             shifts, state, energies, forces, virials, control);
    }
}

void compute_adp(const AdpPotential& potential, std::int64_t n, const std::int32_t* species,
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, const Control& control) {
    // F'(rho), the dipole and the quadrupole (xx yy zz yz xz xy) of each atom
    std::vector<double> state(kState * static_cast<std::size_t>(n), 0.0);
    adp_densities(potential, n, species, positions, matrix, offsets, neighbors, shifts,
                  energies, state.data(), stage(control, 0, 2));
    adp_forces(potential, n, species, positions, matrix, offsets, neighbors, shifts,
               state.data(), energies, forces, virials, stage(control, 1, 2));
}

}  // namespace mdapy::calculators

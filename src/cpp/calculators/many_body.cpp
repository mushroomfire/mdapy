// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/many_body.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "neighbor/frame.h"

namespace mdapy::calculators {

std::vector<std::int64_t> reverse_pairs(std::int64_t n, const std::int64_t* offsets,
                                        const index_t* neighbors, const std::int8_t* shifts,
                                        const Control& control) {
    const std::int64_t count = offsets[n];
    // each atom's pairs ordered by neighbor and shift
    std::vector<std::int64_t> order(static_cast<std::size_t>(count));
    const auto before = [&](std::int64_t a, std::int64_t b) {
        if (neighbors[a] != neighbors[b]) {
            return neighbors[a] < neighbors[b];
        }
        for (int e = 0; e < 3; ++e) {
            if (shifts[3 * a + e] != shifts[3 * b + e]) {
                return shifts[3 * a + e] < shifts[3 * b + e];
            }
        }
        return false;
    };
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                std::iota(order.begin() + offsets[i], order.begin() + offsets[i + 1], offsets[i]);
                std::sort(order.begin() + offsets[i], order.begin() + offsets[i + 1], before);
            }
        },
        control);
    std::vector<std::int64_t> reverse(static_cast<std::size_t>(count), -1);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const std::int64_t j = neighbors[p];
                    const std::int64_t key[4] = {i, -shifts[3 * p], -shifts[3 * p + 1],
                                                 -shifts[3 * p + 2]};
                    const auto below = [&](std::int64_t q, const std::int64_t* k) {
                        const std::int64_t mine[4] = {neighbors[q], shifts[3 * q],
                                                      shifts[3 * q + 1], shifts[3 * q + 2]};
                        return std::lexicographical_compare(mine, mine + 4, k, k + 4);
                    };
                    const auto first = order.begin() + offsets[j];
                    const auto last = order.begin() + offsets[j + 1];
                    const auto found = std::lower_bound(first, last, key, below);
                    if (found != last && neighbors[*found] == i &&
                        shifts[3 * *found] == key[1] && shifts[3 * *found + 1] == key[2] &&
                        shifts[3 * *found + 2] == key[3]) {
                        reverse[p] = *found;
                    }
                }
            }
        },
        control);
    if (std::find(reverse.begin(), reverse.end(), -1) != reverse.end()) {
        throw std::invalid_argument("the neighbor list is not symmetric");
    }
    return reverse;
}

void gather(const Shares& shares, std::int64_t n, const std::int64_t* offsets,
            const index_t* neighbors, const std::int8_t* shifts, double* energies,
            double* forces, double* virials, const Control& control, bool heat,
            const double* positions, const double* matrix) {
    // the pairs reversed are quick: cancelled, but not in the progress
    const std::vector<std::int64_t> reverse =
        reverse_pairs(n, offsets, neighbors, shifts, {control.cancel, {}});
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const double* own = &shares.own_force[3 * static_cast<std::size_t>(i)];
                double force[3] = {own[0], own[1], own[2]};
                double energy = shares.own_energy[static_cast<std::size_t>(i)];
                double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (!heat) {
                    for (int k = 0; k < 9; ++k) {
                        w[k] = shares.own_virial[9 * static_cast<std::size_t>(i) + static_cast<std::size_t>(k)];
                    }
                }
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const auto q = static_cast<std::size_t>(reverse[static_cast<std::size_t>(p)]);
                    const double* t_ij = &shares.derivative[3 * static_cast<std::size_t>(p)];
                    const double* t_ji = &shares.derivative[3 * q];
                    for (int a = 0; a < 3; ++a) {
                        force[a] += t_ij[a] - t_ji[a];
                    }
                    energy += shares.energy[q];
                    if (heat) {
                        // r_ij (x) dU_j/dr_i, the derivative of the terms of j
                        // by r_ji = x_i - x_j being that by x_i
                        double r[3];
                        neighbor::pair_vector(positions + 3 * i,
                                              positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                                              shifts + 3 * p, matrix, r);
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                w[3 * a + b] += r[a] * t_ji[b];
                            }
                        }
                    } else {
                        for (int k = 0; k < 9; ++k) {
                            w[k] += shares.virial[9 * q + static_cast<std::size_t>(k)];
                        }
                    }
                }
                energies[i] = energy;
                for (int a = 0; a < 3; ++a) {
                    forces[3 * i + a] = force[a];
                }
                for (int k = 0; k < 9; ++k) {
                    virials[9 * i + k] = w[k];
                }
            }
        },
        control);
}

}  // namespace mdapy::calculators

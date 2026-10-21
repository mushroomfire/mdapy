// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Neighbor search with cells, for any triclinic box and any combination of
// periodic and non-periodic directions. Periodic images are enumerated, so a
// cutoff larger than the box is allowed: an atom then sees several images of
// the same neighbor, and images of itself.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::neighbor {

// The pairs (i, j, shift) with |x_j - x_i + shift . H| < cutoff, where H has
// the lattice vectors as rows and shift is an integer vector (zero along
// non-periodic directions); the pair (i, i, 0) is excluded. Pairs are
// grouped by i in CSR form, and sorted by j and then shift within a group,
// so the result does not depend on the number of threads.
struct NeighborPairs {
    std::vector<std::int64_t> offsets;    // n + 1: pairs of atom i are [offsets[i], offsets[i + 1])
    std::vector<std::int32_t> neighbors;  // j of each pair
    std::vector<std::int32_t> shifts;     // 3 per pair
    std::vector<double> distances;        // |r_ij| of each pair
};

// positions: n x 3; matrix: 3 x 3, rows a, b, c; origin: 3; pbc: 3.
// Throws std::invalid_argument for a non-positive cutoff or a singular box.
NeighborPairs build(const double* positions, std::int64_t n, const double* matrix,
                    const double* origin, const bool* pbc, double cutoff,
                    const Control& control = {});

}  // namespace mdapy::neighbor

// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Steinhardt bond-orientational order parameters.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::analysis {

constexpr int kMaxDegree = 20;

// Real orthonormal spherical harmonics of the unit vector u for degree l up
// to kMaxDegree, m = 0, then the cosine and sine parts of m = 1..l: 2 l + 1
// values.
void real_harmonics(int l, const double* u, double* y);

// The order parameters q_l of n atoms for each degree in degrees, from the
// vectors to their neighbors in CSR form (neighbors gives the atom of each
// pair), each pair weighted by weights, or all alike when weights is null.
// With average, the harmonics of each atom are averaged with those of its
// neighbors first. q is (n, degrees).
void steinhardt(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                const double* vectors, const double* weights, const std::vector<int>& degrees,
                bool average, double* q, const Control& control = {});

// steinhardt without averaging, with the k nearest neighbors of every atom
// found by a k-d tree as it goes, without a list (positions: n x 3; matrix:
// rows a, b, c; origin: 3; pbc: 3). Throws as a search of the k nearest
// neighbors does.
void steinhardt_nearest(const double* positions, std::int64_t n, const double* matrix,
                        const double* origin, const bool* pbc, int k,
                        const std::vector<int>& degrees, double* q, const Control& control = {});

}  // namespace mdapy::analysis

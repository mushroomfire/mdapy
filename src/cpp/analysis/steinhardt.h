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
// pair). With average, the harmonics of each atom are averaged with those of
// its neighbors first. q is (n, degrees).
void steinhardt(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                const double* vectors, const std::vector<int>& degrees, bool average, double* q,
                const Control& control = {});

}  // namespace mdapy::analysis

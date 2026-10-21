// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Histograms of pair distances, counted during the neighbor search without
// storing the pairs.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::analysis {

// Counts of the ordered pairs (i, j) closer than r_max in bins of equal width,
// by the types of i and j: counts[(type_i * types + type_j) * bins + bin].
// positions: n x 3; matrix: rows a, b, c; types: n values from 0 to types - 1.
std::vector<std::int64_t> pair_histogram(const double* positions, std::int64_t n,
                                         const double* matrix, const double* origin,
                                         const bool* pbc, double r_max, int bins,
                                         const std::int32_t* type, int types,
                                         const Control& control = {});

}  // namespace mdapy::analysis

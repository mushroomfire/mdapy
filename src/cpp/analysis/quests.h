// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The descriptor of QUESTS (Schwalbe-Koda, Hamel, Sadigh, Zhou and Lordi,
// Nat. Commun. 16, 4014, 2025): for each atom, its k nearest neighbors
// within a cutoff, as the weight of each over its distance, and the mean of
// the sorted cross terms of each neighbor with the others.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::analysis {

// The descriptor of each of n atoms (n x (2k - 1)): first, for its k
// nearest neighbors closer than cutoff, nearest first with ties by atom and
// image, w(r) / r with w(r) = (1 - (r / cutoff)^2)^2, zero for missing
// neighbors; then, for s = 0 .. k - 2, the sum over these neighbors j of
// the (s + 1)th largest of sqrt(w_j w_l) / r_jl over the other neighbors
// l, divided by k. positions n x 3; matrix, the rows a, b, c of the box;
// the neighbors of each atom in CSR form with their periodic images
// (offsets n + 1, others and shifts 3 each), the atom itself left out.
std::vector<double> quests(int k, double cutoff, std::int64_t n, const double* positions,
                           const double* matrix, const std::int64_t* offsets,
                           const index_t* others, const std::int8_t* shifts,
                           const Control& control = {});

// For each of m points x (m x dimension), log sum_j exp(-|x - y_j|^2 /
// (2 bandwidth^2)) over count points y (count x dimension), taken from the
// nearest y so that it stays finite far from all of them.
std::vector<double> log_gaussian_sums(std::int64_t m, const double* x, std::int64_t count,
                                      const double* y, std::int64_t dimension, double bandwidth,
                                      const Control& control = {});

}  // namespace mdapy::analysis

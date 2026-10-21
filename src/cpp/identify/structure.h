// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Identification of local crystal structure: the centro-symmetry parameter
// and common neighbor analysis.

#pragma once

#include <cstdint>

#include "core/parallel.h"

namespace mdapy::identify {

// Structure types of common neighbor analysis.
enum Structure : std::int8_t { kOther = 0, kFcc = 1, kHcp = 2, kBcc = 3, kIco = 4 };

// The centro-symmetry parameter of n atoms from the vectors (n x k x 3) to
// their k neighbors, k even: the sum of the k / 2 smallest |r_a + r_b|^2
// over the pairs of neighbors a < b.
void centro_symmetry(std::int64_t n, int k, const double* vectors, double* csp,
                     const Control& control = {});

// Common neighbor analysis of n atoms from the vectors to their neighbors in
// CSR form. With cutoff > 0 the neighbors are those within the cutoff;
// otherwise the cutoff of each atom is adaptive and the neighbors must be
// sorted by distance, at least 14 of them for bcc to be found.
void common_neighbor_analysis(std::int64_t n, const std::int64_t* offsets, const double* vectors,
                              double cutoff, std::int8_t* types, const Control& control = {});

// Diamond structures: 0 other, 1 cubic, 2 hexagonal.
enum Diamond : std::int8_t { kNotDiamond = 0, kCubicDiamond = 1, kHexagonalDiamond = 2 };

// Cubic or hexagonal diamond for each of n atoms from its four nearest
// neighbors (neighbors and vectors, n x 4): the twelve second neighbors,
// reached through the first ones, must form fcc (cubic) or hcp (hexagonal)
// under adaptive common neighbor analysis.
void identify_diamond(std::int64_t n, const std::int32_t* neighbors, const double* vectors,
                      std::int8_t* types, const Control& control = {});

}  // namespace mdapy::identify

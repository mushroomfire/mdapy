// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Atomic strain: the deformation gradient that maps each atom's neighbor
// vectors in a reference configuration onto the current ones.

#pragma once

#include <cstdint>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::analysis {

// For n atoms with reference neighbors in CSR form (neighbor and integer
// shift of each pair): reference vectors (S_j + shift - S_i) H_ref and
// current vectors (S_j + D_j + shift - S_i - D_i) H, with S the reference
// fractional coordinates and D the fractional displacements (n x 3 each).
// Gives the Green-Lagrange strain (n x 9) and the non-affine squared
// displacement (n); NaN where the neighbors do not span three dimensions.
void atomic_strain(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                   const std::int8_t* shifts, const double* reference, const double* displacement,
                   const double* reference_matrix, const double* matrix, double* strain,
                   double* nonaffine, const Control& control = {});

}  // namespace mdapy::analysis

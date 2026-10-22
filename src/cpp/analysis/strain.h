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

// What the atomic strain of every atom needs besides its reference
// neighbors: the reference fractional coordinates S (n x 3) and the current
// positions x (n x 3) with the inverse (3 x 3, x . inverse gives fractional
// coordinates) and origin of the current box, its periodic directions, and
// the matrices that turn fractional vectors into reference and current
// vectors (the current box, or the reference box for an affine mapping).
struct StrainInput {
    std::int64_t n;
    const double* reference;
    const double* positions;
    const double* inverse;
    const double* origin;
    const bool* pbc;
    const double* reference_matrix;
    const double* matrix;
};

// The results for n atoms: the Green-Lagrange strain (n x 9), its von Mises
// shear invariant and a third of its trace (n), and the non-affine squared
// displacement (n); NaN where the neighbors do not span three dimensions.
struct StrainOutput {
    double* strain;
    double* shear;
    double* volumetric;
    double* nonaffine;
};

// The atomic strain with the neighbors in the reference closer than cutoff,
// found on a cell grid of the reference positions (n x 3) in the reference
// box (reference_matrix, origin, pbc).
void atomic_strain_search(const double* reference_positions, const double* reference_origin,
                          double cutoff, const StrainInput& input, const StrainOutput& output,
                          const Control& control = {});

// The same with the neighbors of a list of the reference in CSR form, of
// which those closer than cutoff count.
void atomic_strain_list(const std::int64_t* offsets, const index_t* neighbors,
                        const std::int8_t* shifts, const double* distances, double cutoff,
                        const StrainInput& input, const StrainOutput& output,
                        const Control& control = {});

}  // namespace mdapy::analysis

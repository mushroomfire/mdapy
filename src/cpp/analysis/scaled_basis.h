// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// A descriptor of the neighborhood of each atom in a fixed basis, on
// distances scaled by the radii of each pair, for the generation of
// training structures: radial Chebyshev functions as those of NEP, angular
// sums of Legendre polynomials, and radial sums for each element.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::analysis {

struct ScaledBasis {
    int radial = 8;        // radial functions, n < radial
    int angular = 4;       // radial functions of the angular sums, n < angular
    int l_max = 4;         // degrees 1 .. l_max of the angular sums
    int chemical = 2;      // radial functions summed for each element when there are several
    double cutoff = 2.4;   // of the scaled distance s = r / (R_i + R_j), radial
    double angular_cutoff = 1.7;  // the same, angular
};

// The number of values per atom for species elements.
std::int64_t scaled_basis_dimension(const ScaledBasis& basis, int species);

// The descriptor of each of n atoms (n x dimension): for f_k(s) = (T_k(2
// (s / c - 1)^2 - 1) + 1) / 2 fc(s), fc(s) = (1 + cos(pi s / c)) / 2, the
// sums over the neighbors of f_n, n < radial; the sums over pairs of
// neighbors j, k (j = k too) of f_n(s_j) f_n(s_k) P_l(cos theta_jk) with the
// angular cutoff, n < angular, 1 <= l <= l_max; and with several species,
// the sums of f_n, n < chemical, over the neighbors of each species. species,
// the index of the element of each atom; radii, of each element; positions
// n x 3; matrix, the rows a, b, c of the box; the neighbors in CSR form with
// their periodic images, the atom itself left out.
std::vector<double> scaled_basis(const ScaledBasis& basis, int species_count,
                                 const std::int32_t* species, const double* radii,
                                 std::int64_t n, const double* positions, const double* matrix,
                                 const std::int64_t* offsets, const index_t* others,
                                 const std::int8_t* shifts, const Control& control = {});

}  // namespace mdapy::analysis

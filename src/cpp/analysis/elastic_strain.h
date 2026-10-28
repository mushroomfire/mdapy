// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Elastic strain: the deformation of the neighbors of each atom of a crystal
// from those of the ideal crystal, the neighbors in the order the structure
// identification of the dislocation analysis gives them.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::analysis {

// The crystal (a PTM type); the length in Å of the cubic lattice constant
// of its ideal vectors, for hcp and hexagonal diamond that of the cubic
// crystal with the same nearest neighbors; and the stretch of the ideal
// vectors along c, 1 for the cubic crystals.
struct ElasticStrainOptions {
    int crystal = 0;
    double a = 1.0;
    double axial = 1.0;
};

// Of each atom, its structure (a PTM type, 0 for other) and, of the atoms of
// the crystal, the strain in the spatial frame, (I - F^-T F^-1) / 2 of the
// deformation gradient F that maps the ideal vectors onto the neighbors by
// least squares (9 each); 0 for the other atoms, NaN where F is singular.
struct ElasticStrain {
    std::vector<std::int8_t> structure;
    std::vector<double> strain;
};

// Positions: n x 3 in the box along its periodic directions; matrix: rows
// a, b, c; inverse: its inverse; origin: 3; pbc: 3. Throws as the
// structure identification does, and std::invalid_argument where an atom of
// the crystal meets its own periodic image among its neighbors.
ElasticStrain elastic_strain(const double* positions, std::int64_t n, const double* matrix,
                             const double* inverse, const double* origin, const bool* pbc,
                             const ElasticStrainOptions& options, const Control& control = {});

}  // namespace mdapy::analysis

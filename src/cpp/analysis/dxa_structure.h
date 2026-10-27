// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The structures the dislocation analysis works in: crystal atoms by common
// neighbor analysis, their neighbors in the order of the template of their
// lattice.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::analysis::dxa {

// The PTM type of each atom, 0 for other, and its neighbors in the order of
// the first template of its lattice (16 each, -1 after the last).
struct Structures {
    std::vector<std::int8_t> structure;
    std::vector<index_t> neighbors;
};

// The atoms of crystal (a PTM type) and of the structure its planar defects
// make, when defects is set: for fcc or hcp, fcc and hcp atoms by common
// neighbor analysis of their 12 nearest neighbors with the cutoff halfway
// between the first two shells of their mean distance, the 13th neighbor
// beyond it; for bcc, bcc atoms by that of their 14 nearest, the cutoff
// from the mean distance of the 8 nearest, the 15th beyond it; for cubic or
// hexagonal diamond, both from the 4 nearest neighbors and the 3 nearest of
// each of those other than the atom, the cutoff from the mean distance of
// the 12 (positions: n x 3; matrix: rows a, b, c; origin: 3; pbc: 3).
// Throws as a search of nearest neighbors does.
Structures find_structures(const double* positions, std::int64_t n, const double* matrix,
                           const double* origin, const bool* pbc, int crystal, bool defects,
                           const Control& control = {});

}  // namespace mdapy::analysis::dxa

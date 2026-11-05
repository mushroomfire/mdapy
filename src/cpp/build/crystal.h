// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The atoms of a perfect crystal: copies of a cell, in order.

#pragma once

#include <cstdint>

#include "core/buffer.h"
#include "core/parallel.h"

namespace mdapy::build {

struct CrystalFill {
    const std::int64_t* counts = nullptr;  // (3) cells along each lattice vector
    const double* vectors = nullptr;       // (3, 3) the lattice vectors of a cell, as rows
    std::int64_t planes = 0;               // of a cell, along the third vector
    const std::int64_t* plane_rows = nullptr;  // (planes + 1) the rows of each plane
    const std::int64_t* row_atoms = nullptr;   // (rows + 1) the atoms of each row
    const double* fractional = nullptr;        // (atoms, 3) of a cell, by row
    const std::int64_t* kinds = nullptr;       // (atoms) of a cell, by row
};

struct CrystalAtoms {
    Buffer<double> positions;    // 3 per atom
    Buffer<std::uint32_t> kind;  // of each atom
};

// The atoms of the crystal, by cell along the third vector, then by plane of a
// cell, then by cell along the second vector, then by row of the plane, then by
// cell along the first vector, then by atom of the row.
CrystalAtoms fill_crystal(const CrystalFill& fill, const Control& control = {});

}  // namespace mdapy::build

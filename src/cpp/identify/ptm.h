// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Polyhedral template matching (Larsen, Schmidt and Schiøtz 2016), with the
// conventions of OVITO: its structure and ordering numbers, the order of the
// template neighbors and the codes of their correspondences.

#pragma once

#include <cstdint>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::identify {

// The structures PTM tells apart: OVITO's numbers, those of the library.
enum PtmStructure : std::int8_t {
    kPtmOther = 0,
    kPtmFcc = 1,
    kPtmHcp = 2,
    kPtmBcc = 3,
    kPtmIco = 4,
    kPtmSc = 5,
    kPtmCubicDiamond = 6,
    kPtmHexagonalDiamond = 7,
    kPtmGraphene = 8,
};

// The atoms and their nearest neighbors, at most 18 each, nearest first, in
// CSR form; the vector to neighbor j is x[j] - x[i] + shift . matrix.
struct PtmInput {
    std::int64_t n = 0;
    const double* positions = nullptr;      // n x 3
    const double* matrix = nullptr;         // rows a, b, c
    const std::int64_t* offsets = nullptr;  // n + 1
    const index_t* j = nullptr;
    const std::int8_t* shifts = nullptr;    // 3 per pair
    const std::int32_t* types = nullptr;    // n chemical types, or none: no ordering
    std::int32_t structures = 0;            // the bits 1 << (structure - 1) to look for
    bool deformation = false;               // compute deformation gradients
};

// The results of each atom; null pointers are not written. An atom that
// matches no structure gets zeros.
struct PtmOutput {
    std::int8_t* structure = nullptr;          // n
    double* rmsd = nullptr;                    // n
    double* interatomic_distance = nullptr;    // n
    double* orientation = nullptr;             // n x 4: x, y, z, w
    double* deformation = nullptr;             // n x 9, row-major
    std::int8_t* ordering = nullptr;           // n
    std::uint64_t* correspondences = nullptr;  // n, as OVITO encodes them
    std::int8_t* template_index = nullptr;     // n
    index_t* neighbors = nullptr;              // n x 16: in the order of the template, -1 after
};

// Matches every atom: first orders the neighbors of each atom by the faces
// of its Voronoi cell, then matches the templates.
void polyhedral_template_matching(const PtmInput& input, const PtmOutput& output,
                                  const Control& control = {});

}  // namespace mdapy::identify

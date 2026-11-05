// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The atoms of the grains of a polycrystal: copies of a unit, rotated, that
// fill the Voronoi cell of each seed.

#pragma once

#include <cstdint>

#include "core/buffer.h"
#include "core/parallel.h"

namespace mdapy::build {

struct GrainFill {
    std::int64_t grains = 0;
    const double* seeds = nullptr;        // (grains, 3)
    const double* rotations = nullptr;    // (grains, 3, 3), taking the unit into each grain
    const double* radii = nullptr;        // (grains) no atom of a grain is farther from its seed
    const std::int64_t* spans = nullptr;  // (grains, 3) copies of the unit on either side of a seed
    const std::int64_t* offsets = nullptr;  // (grains + 1) the faces of grain g
    const double* faces = nullptr;          // (faces, 3) from a seed to the seed across a face
    const double* cell = nullptr;           // (3, 3) the lattice vectors of the unit, as rows
    const double* origin = nullptr;         // (3) the point of the unit placed at the seed
    std::int64_t sites = 0;
    const double* site_positions = nullptr;  // (sites, 3) fractional coordinates
    const std::int64_t* centers_of = nullptr;  // (sites) the center that decides each site
    std::int64_t centers = 0;
    const double* center_positions = nullptr;  // (centers, 3) fractional coordinates
    int axis = -1;  // of a columnar polycrystal: distances across it only; -1 for none
    double tolerance = 0.0;  // a center belongs to a grain up to this much farther from its seed
    bool copies = false;  // also give the copy of the unit of each atom
    double margin = -1.0;  // from 0 on: also tell the atoms this close to a face, in Å
};

struct GrainAtoms {
    Buffer<double> positions;     // 3 per atom, not wrapped
    Buffer<std::int64_t> grain;   // of each atom
    Buffer<std::int64_t> site;    // of the unit, for each atom
    Buffer<std::int64_t> copy;    // with copies: the copies that hold atoms, numbered from 0
    Buffer<std::uint8_t> edge;    // with a margin: 1 for an atom within it of a face of its grain
};

// The atoms of every grain, by grain, then by copy of the unit (the indices of
// its lattice vectors in increasing order, the last the fastest), then by site.
// A site is in a grain when its center is no farther from the seed of the grain
// than from any other seed, up to the tolerance, which is what the faces give.
GrainAtoms fill_grains(const GrainFill& fill, const Control& control = {});

}  // namespace mdapy::build

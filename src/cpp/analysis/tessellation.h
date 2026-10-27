// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The Delaunay tessellation of atoms from their Voronoi cells.

#pragma once

#include <cstdint>
#include <vector>

#include "analysis/voronoi.h"
#include "core/parallel.h"

namespace mdapy::analysis {

// The tetrahedra, each once: the atoms at their corners, the lowest first
// and the others in increasing order (4 each), the periodic image of each
// (12 each, the first at 0), the tetrahedron across the face opposite each
// corner (4 each, -1 for none) and the corner of that one opposite the face
// (4 each), and the radius of the sphere through the corners (Å). Vertices
// on the walls of a box that is not periodic give none; repeated tells
// whether an atom met its own periodic image, whose tetrahedra are left out.
struct Tetrahedra {
    std::vector<std::int64_t> atoms;
    std::vector<std::int8_t> shifts;
    std::vector<std::int64_t> neighbors;
    std::vector<std::int8_t> sides;
    std::vector<double> radii;
    bool repeated = false;
};

// The tetrahedra of the vertices of Voronoi cells.
Tetrahedra delaunay(const VoronoiVertices& cells, const Control& control = {});

}  // namespace mdapy::analysis

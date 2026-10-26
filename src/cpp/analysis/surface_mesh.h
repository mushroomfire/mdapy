// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The surface mesh of atoms from the alpha shape of their Delaunay
// tessellation, and the solid and empty regions it separates.

#pragma once

#include <cstdint>
#include <vector>

#include "analysis/tessellation.h"
#include "core/parallel.h"

namespace mdapy::analysis {

// The tetrahedra whose circumscribed sphere is smaller than the probe
// sphere are solid (1), the others and the space outside the tessellation
// empty (0). volumes: of each tetrahedron (Å^3). The surface between them:
// the atom of each vertex in increasing order and its position, the
// vertices of each triangle (3 each, counterclockwise seen from the empty
// side), the periodic image of each corner (9 each, the first at 0) and
// the solid and the empty tetrahedron on its two sides (2 each, -1 outside
// the tessellation).
struct SurfaceMesh {
    std::vector<std::uint8_t> solid;
    std::vector<double> volumes;
    std::vector<std::int64_t> atoms;
    std::vector<double> vertices;
    std::vector<std::int64_t> faces;
    std::vector<std::int8_t> shifts;
    std::vector<std::int64_t> cells;
};

// The alpha shape of n atoms at positions (n x 3) of their tetrahedra,
// which were found from the atoms at jittered; matrix: rows a, b, c.
SurfaceMesh alpha_shape(const Tetrahedra& tetrahedra, std::int64_t n, const double* positions,
                        const double* jittered, const double* matrix, double radius,
                        const Control& control = {});

// Moves the vertices of the mesh by iterations steps of Taubin's smoothing,
// which keeps the volume enclosed.
void smooth_mesh(SurfaceMesh& mesh, const double* matrix, int iterations,
                 const Control& control = {});

// The regions of connected solid or empty tetrahedra: the region of each
// tetrahedron, the solid one and the empty one on the two sides of each
// triangle of the mesh (2 each), and for each region whether it is solid,
// is the empty space outside the tessellation (exterior), wraps around a
// periodic boundary (periodic), and the volume of its tetrahedra (Å^3).
// The space outside the tessellation and the empty tetrahedra next to it
// form the first region, when there is one.
struct Regions {
    std::vector<std::int64_t> region;
    std::vector<std::int64_t> faces;
    std::vector<std::uint8_t> solid, exterior, periodic;
    std::vector<double> volumes;
};

Regions surface_regions(const Tetrahedra& tetrahedra, const SurfaceMesh& mesh);

}  // namespace mdapy::analysis

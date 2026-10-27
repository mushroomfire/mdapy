// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The dislocation analysis of atoms, the dislocation extraction algorithm
// (Stukowski, Bulatov and Arsenlis 2012): the crystal atoms and their
// clusters, the interface mesh between the good crystal and the rest, and
// the dislocation lines Burgers circuits on it find.

#pragma once

#include <cstdint>

#include "analysis/dxa_clusters.h"
#include "analysis/dxa_structure.h"
#include "analysis/dxa_tracer.h"
#include "analysis/surface_mesh.h"
#include "core/parallel.h"

namespace mdapy::analysis {

struct DislocationOptions {
    int crystal = 0;          // the PTM type of the crystal
    bool defects = true;      // its planar defects too, for partial dislocations
    int max_circuit = 14;     // the longest Burgers circuit that finds a line
    int stretch = 9;          // the edges more a circuit may grow along a line
    double separation = 2.5;  // of the points of the lines, 0 to keep them all
    int smoothing = 1;        // steps of smoothing of the lines
    int mesh_smoothing = 8;   // steps of smoothing of the defect mesh
    bool interface = false;   // also the interface mesh
};

// The structures and clusters of the atoms, the dislocation lines, the
// mesh around the defects that are not dislocations and, when asked for,
// the interface mesh between the good crystal and the rest: of each, the
// atom at each vertex (-1 at the end of a line), the positions of the
// vertices, the vertices of each triangle and the periodic image of each
// corner.
struct Dislocations {
    dxa::Structures structures;
    dxa::Clusters clusters;
    dxa::Network network;
    SurfaceMesh defects;
    SurfaceMesh interface;
};

// The dislocations of n atoms at positions (n x 3) whose tessellation comes
// from jittered, the same moved at random by a little, both inside the box
// along its periodic directions (matrix: rows a, b, c; inverse; origin: 3;
// pbc: 3). Throws std::invalid_argument for a crystal the analysis does not
// know and for a box too thin along a periodic direction.
Dislocations dislocation_analysis(const double* positions, const double* jittered, std::int64_t n,
                                  const double* matrix, const double* inverse,
                                  const double* origin, const bool* pbc,
                                  const DislocationOptions& options, const Control& control = {});

}  // namespace mdapy::analysis

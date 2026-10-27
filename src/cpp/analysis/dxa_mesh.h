// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The interface mesh of the dislocation analysis: the triangles between the
// good tetrahedra, where the elastic mapping is compatible, and the rest,
// with the ideal vector of each edge, on which Burgers circuits run.

#pragma once

#include <cstdint>
#include <vector>

#include "analysis/dxa_clusters.h"
#include "analysis/dxa_mapping.h"
#include "analysis/tessellation.h"
#include "core/parallel.h"

namespace mdapy::analysis::dxa {

// The triangles, counterclockwise seen from the good side, as half-edges:
// e = 3 f + k runs from corner k of triangle f to corner k + 1 (mod 3). A
// vertex is an atom; an atom where good regions touch only at that point
// gives a vertex for each, so that the triangles around every vertex make
// one disk.
struct InterfaceMesh {
    // of each vertex: its atom and position (Å)
    std::vector<std::int64_t> atoms;
    std::vector<double> positions;  // 3 each
    // of each half-edge: the vertex it starts at, the half-edge the other
    // way, the vector between its atoms (Å), the ideal vector in the frame
    // of the cluster of its start, that cluster, and the matrix that takes
    // vectors of the frame of the cluster of its end into that of its start
    std::vector<std::int64_t> start;
    std::vector<std::int64_t> opposite;
    std::vector<double> physical;  // 3 each
    std::vector<double> ideal;     // 3 each
    std::vector<std::int64_t> cluster;
    std::vector<std::int32_t> back;  // into matrices; 0, the identity, within a cluster
    std::vector<Mat> matrices;
    // the half-edges from each vertex
    std::vector<std::int64_t> offsets, edges;

    std::int64_t vertices() const { return static_cast<std::int64_t>(atoms.size()); }
    std::int64_t faces() const { return static_cast<std::int64_t>(start.size()) / 3; }
    std::int64_t end(std::int64_t e) const { return start[static_cast<std::size_t>(next(e))]; }
    static std::int64_t face(std::int64_t e) { return e / 3; }
    static std::int64_t next(std::int64_t e) { return 3 * (e / 3) + (e % 3 + 1) % 3; }
    static std::int64_t prev(std::int64_t e) { return 3 * (e / 3) + (e % 3 + 2) % 3; }
};

// The largest neighbor cutoff of the crystal atoms (Å): the mean distance
// of their nearest neighbors, the second for diamond, scaled to halfway to
// the next shell; 0 without crystal atoms.
double neighbor_cutoff(const ClusterInput& atoms);

// The mesh between the tetrahedra where good is set and whose
// circumscribed sphere has a squared radius below alpha (Å^2) and the
// rest; jittered: the positions the tetrahedra were found from. Throws
// std::runtime_error when an edge spans half of a periodic box or more.
InterfaceMesh interface_mesh(const ClusterInput& atoms, const Mapping& mapping,
                             const Tetrahedra& tetrahedra, const std::vector<std::uint8_t>& good,
                             const double* jittered, double alpha, const Control& control = {});

}  // namespace mdapy::analysis::dxa

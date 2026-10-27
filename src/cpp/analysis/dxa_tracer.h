// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The dislocation lines of the interface mesh: Burgers circuits that do
// not close in the ideal crystal, found shortest first, swept along the
// lines in both directions and joined where they meet.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "analysis/dxa_clusters.h"
#include "analysis/dxa_mesh.h"
#include "core/parallel.h"

namespace mdapy::analysis::dxa {

// A dislocation line between two ends: its points (Å, unwrapped, the
// consecutive ones close), the number of edges of the Burgers circuit at
// each, the Burgers vector in the frame of a cluster, and the junction at
// each end, the first and the last (-1 for a free end); closed for a loop
// or a line through periodic boundaries, which has no ends.
struct Segment {
    std::vector<double> points;  // 3 each
    std::vector<std::int32_t> cores;
    Vec burgers{};
    std::int64_t cluster = -1;
    std::array<std::int64_t, 2> junctions{-1, -1};
    bool closed = false;
};

// A cap on the circuit at a free end of a line: its end and the edges of
// the circuit.
struct Cap {
    Vec apex{};
    std::vector<std::int64_t> edges;
};

// The lines, the number of junctions, and the mesh around the defects that
// are not dislocations: the triangles of the interface mesh no circuit took
// for good, and the caps.
struct Network {
    std::vector<Segment> segments;
    std::int64_t junctions = 0;
    std::vector<std::int64_t> faces;
    std::vector<Cap> caps;
};

// The lines of the mesh: circuits of at most max_circuit edges find them,
// and stretch more edges let a circuit pass on along a line. A Burgers
// vector in the frame of a cluster of a structure other than the crystal
// is given in that of its crystal neighbor touching it most.
Network trace_dislocations(const InterfaceMesh& mesh, const ClusterInput& atoms,
                           const Clusters& clusters, int max_circuit, int stretch,
                           const Control& control = {});

// Fewer points along each line, each the mean of a run of them about
// separation times as long as the circuits there have edges, then levels
// steps of Taubin's smoothing; 0 skips either. The points keep their core
// sizes only without the first.
void smooth_lines(Network& network, double separation, int levels);

}  // namespace mdapy::analysis::dxa

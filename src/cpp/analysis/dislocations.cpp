// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/dislocations.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "analysis/dxa_mapping.h"
#include "analysis/dxa_mesh.h"
#include "analysis/tessellation.h"
#include "analysis/voronoi.h"

namespace mdapy::analysis {

Dislocations dislocation_analysis(const double* positions, const double* jittered, std::int64_t n,
                                  const double* matrix, const double* inverse,
                                  const double* origin, const bool* pbc,
                                  const DislocationOptions& options, const Control& control) {
    if (dxa::lattice(options.crystal) == nullptr) {
        throw std::invalid_argument("no lattice for this structure");
    }
    constexpr int kStages = 5;
    Dislocations out;
    out.structures = dxa::find_structures(positions, n, matrix, origin, pbc, options.crystal,
                                          options.defects, stage(control, 0, kStages));
    dxa::ClusterInput input;
    input.n = n;
    input.structure = out.structures.structure.data();
    input.neighbors = out.structures.neighbors.data();
    input.positions = positions;
    input.matrix = matrix;
    input.inverse = inverse;
    input.pbc = pbc;
    input.crystal = options.crystal;
    out.clusters = dxa::build_clusters(input);
    if (out.clusters.structure.empty()) {
        return out;
    }
    // tetrahedra with a circumscribed sphere wider than five times the
    // largest neighbor cutoff, in Å^2, are empty space; the walls of the
    // open directions lie beyond the reach of such a sphere
    const double alpha = 5 * dxa::neighbor_cutoff(input);
    double walls[9], corner[3];
    std::copy(matrix, matrix + 9, walls);
    std::copy(origin, origin + 3, corner);
    for (int axis = 0; axis < 3; ++axis) {
        if (pbc[axis]) {
            continue;
        }
        double spacing = 0.0;  // between the faces of the box across the axis
        for (int d = 0; d < 3; ++d) {
            spacing += inverse[3 * d + axis] * inverse[3 * d + axis];
        }
        spacing = 1.0 / std::sqrt(spacing);
        double low = 0.0, high = 0.0;
        for (std::int64_t i = 0; i < n; ++i) {
            double f = 0.0;
            for (int d = 0; d < 3; ++d) {
                f += (jittered[3 * i + d] - origin[d]) * inverse[3 * d + axis];
            }
            low = i == 0 ? f : std::min(low, f);
            high = i == 0 ? f : std::max(high, f);
        }
        const double margin = (std::sqrt(alpha) + 1.0) / spacing;
        low -= margin;
        high += margin;
        for (int d = 0; d < 3; ++d) {
            corner[d] += low * matrix[3 * axis + d];
            walls[3 * axis + d] = (high - low) * matrix[3 * axis + d];
        }
    }
    const Tetrahedra tetrahedra =
        delaunay(voronoi_vertices(jittered, n, walls, corner, pbc, stage(control, 1, kStages)),
                 stage(control, 2, kStages));
    if (tetrahedra.repeated) {
        throw std::invalid_argument("an atom meets its own periodic image");
    }
    const dxa::Mapping mapping(input, out.clusters, tetrahedra);
    const auto count = static_cast<std::int64_t>(tetrahedra.radii.size());
    std::vector<std::uint8_t> good(static_cast<std::size_t>(count));
    parallel_for(
        count, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                good[static_cast<std::size_t>(t)] = mapping.good(t) ? 1 : 0;
            }
        },
        stage(control, 3, kStages));
    const dxa::InterfaceMesh mesh =
        dxa::interface_mesh(input, mapping, tetrahedra, good, jittered, alpha);
    out.network = dxa::trace_dislocations(mesh, input, out.clusters, options.max_circuit,
                                          options.stretch, stage(control, 4, kStages));
    dxa::smooth_lines(out.network, options.separation, options.smoothing);
    return out;
}

}  // namespace mdapy::analysis

// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Voronoi cells of atoms, by cutting a polyhedron with the bisecting planes
// of the neighbors in order of distance.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::analysis {

// Faces that count: those larger than face (Å²) and than relative_face times
// the surface of their cell, with three or more edges at least edge (Å) long.
struct VoronoiThresholds {
    double face = 0.0;
    double relative_face = 0.0;
    double edge = 0.0;
};

struct VoronoiCells {
    std::vector<double> volumes;  // (n)
    std::vector<double> radii;    // (n) distance of the farthest vertex
    // the faces of each atom that count, in CSR form: the atom across each
    // face (-1 for a wall of the box) and its periodic image, the area and
    // the number of edges that count; and the number of these faces shared
    // with atoms
    std::vector<std::int64_t> offsets;  // (n + 1)
    std::vector<index_t> neighbors;
    std::vector<std::int8_t> shifts;  // 3 per face
    std::vector<double> areas;
    std::vector<std::int32_t> edges;
    std::vector<std::int64_t> coordination;  // (n)
};

// Voronoi cells of n atoms, with their neighbors searched on a cell grid out
// to a reach that grows for the cells it cannot close. Directions that are
// not periodic are closed by the planes of the box, moved out to include
// every atom. Throws std::invalid_argument for two atoms at the same
// position, std::overflow_error for a face with a periodic image outside int8.
VoronoiCells voronoi(const double* positions, std::int64_t n, const double* matrix,
                     const double* origin, const bool* pbc, const VoronoiThresholds& thresholds,
                     const Control& control = {});

}  // namespace mdapy::analysis

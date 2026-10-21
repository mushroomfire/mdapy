// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Voronoi cells of atoms, by cutting a polyhedron with the bisecting planes
// of the neighbors in order of distance.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::analysis {

struct VoronoiCells {
    std::vector<double> volumes;  // (n)
    std::vector<double> radii;    // (n) distance of the farthest vertex
    // the faces of each atom in CSR form: the atom across each face (-1 for a
    // wall of the box) and its periodic image, the area and the number of
    // vertices; and the number of faces shared with atoms
    std::vector<std::int64_t> offsets;  // (n + 1)
    std::vector<std::int32_t> neighbors;
    std::vector<std::int32_t> shifts;  // 3 per face
    std::vector<double> areas;
    std::vector<std::int32_t> vertices;
    std::vector<std::int64_t> coordination;  // (n)
};

// Voronoi cells of n atoms, with their neighbors searched on a cell grid out
// to a reach that grows for the cells it cannot close. Directions that are
// not periodic are closed by the planes of the box, moved out to include
// every atom. Faces smaller than area_threshold times the surface of their
// cell are left out of the faces, not of the volume.
VoronoiCells voronoi(const double* positions, std::int64_t n, const double* matrix,
                     const double* origin, const bool* pbc, double area_threshold,
                     const Control& control = {});

}  // namespace mdapy::analysis

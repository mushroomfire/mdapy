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
    std::vector<std::uint8_t> complete;  // (n) whether the neighbors sufficed
    // the faces of each atom in CSR form: the pair of the neighbor across
    // the face, or -1 - k for the k-th wall of the box (2 per direction
    // that is not periodic), its area and its number of vertices
    std::vector<std::int64_t> offsets;  // (n + 1)
    std::vector<std::int64_t> pairs;
    std::vector<double> areas;
    std::vector<std::int32_t> vertices;
};

// Voronoi cells of n atoms from the vectors to their neighbors (CSR, every
// neighbor closer than reach). Directions that are not periodic are closed by
// the planes of the box, moved out to include every atom. A cell is complete
// when no atom beyond reach can cut it.
VoronoiCells voronoi(std::int64_t n, const double* positions, const double* matrix,
                     const double* origin, const bool* pbc, const std::int64_t* offsets,
                     const double* vectors, double reach, const Control& control = {});

}  // namespace mdapy::analysis

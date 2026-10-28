// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The surface where a field sampled on a grid crosses a level, by marching
// tetrahedra: the cell between each eight neighboring samples is cut into
// six tetrahedra about its diagonal, which neighboring cells share faces of.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mdapy::analysis {

// The vertices in fractional coordinates of the box (V x 3), in [0, 1)
// along its periodic directions; the triangles (F x 3); and the periodic
// image of each corner of each triangle (F x 3 x 3).
struct Isosurface {
    std::vector<double> vertices;
    std::vector<std::int64_t> faces;
    std::vector<std::int8_t> shifts;
};

// values: n[0] x n[1] x n[2], the last fastest, at the centers of the cells
// of a grid that divides the box along its vectors; pbc: 3. Samples above
// the level are inside; the triangles turn counterclockwise about normals
// that point out, in a right-handed box, or in, when left_handed.
Isosurface isosurface(const double* values, const std::array<std::int64_t, 3>& n,
                      const bool* pbc, double level, bool left_handed);

}  // namespace mdapy::analysis

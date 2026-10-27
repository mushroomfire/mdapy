// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/tessellation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mdapy::analysis {
namespace {

using Shift = std::array<std::int8_t, 3>;

// an atom in one periodic image
struct Corner {
    std::int64_t atom;
    Shift shift;
};

bool before(const Corner& a, const Corner& b) {
    return a.atom < b.atom || (a.atom == b.atom && a.shift < b.shift);
}

bool same(const Corner& a, const Corner& b) {
    return a.atom == b.atom && a.shift == b.shift;
}

class Cells {
public:
    explicit Cells(const VoronoiVertices& v) : v_(v) {}

    // the owner of vertex v (by number) of the cell of atom cell and the
    // atoms across its three faces, these in order; 0 for a tetrahedron, 1
    // at a wall, 2 when an atom repeats
    int corners(std::int64_t cell, std::int64_t v, std::array<Corner, 4>& out) const {
        const auto& chunk = v_.chunks[static_cast<std::size_t>(v_.chunk[cell])];
        const auto at = static_cast<std::size_t>(v_.start[cell] + v - v_.offsets[cell]);
        out[0] = Corner{cell, {0, 0, 0}};
        for (std::size_t k = 0; k < 3; ++k) {
            const auto j = static_cast<std::int64_t>(chunk.atoms[3 * at + k]);
            if (j < 0) {
                return 1;
            }
            const std::int8_t* s = chunk.shifts.data() + 9 * at + 3 * k;
            out[k + 1] = Corner{j, {s[0], s[1], s[2]}};
        }
        for (int k = 1; k < 4; ++k) {
            if (out[k].atom == cell || (k > 1 && out[k].atom == out[k - 1].atom)) {
                return 2;
            }
        }
        return 0;
    }

    double radius(std::int64_t cell, std::int64_t v) const {
        const auto& chunk = v_.chunks[static_cast<std::size_t>(v_.chunk[cell])];
        return chunk.radii[static_cast<std::size_t>(v_.start[cell] + v - v_.offsets[cell])];
    }

    // the vertex of the cell of atom m whose faces lie across the given
    // atoms, -1 for none
    std::int64_t find(std::int64_t m, const std::array<Corner, 3>& wanted) const {
        std::array<Corner, 4> found;
        for (std::int64_t v = v_.offsets[m]; v < v_.offsets[m + 1]; ++v) {
            if (corners(m, v, found) == 1) {
                continue;
            }
            if (same(found[1], wanted[0]) && same(found[2], wanted[1]) &&
                same(found[3], wanted[2])) {
                return v;
            }
        }
        return -1;
    }

    // the other end of the edge of the cell of atom i at vertex v between
    // its faces toward a and b
    std::int64_t across(std::int64_t i, std::int64_t v, const Corner& a, const Corner& b) const {
        std::array<Corner, 4> found;
        for (std::int64_t u = v_.offsets[i]; u < v_.offsets[i + 1]; ++u) {
            if (u == v || corners(i, u, found) == 1) {
                continue;
            }
            bool has_a = false, has_b = false;
            for (int k = 1; k < 4; ++k) {
                has_a = has_a || same(found[k], a);
                has_b = has_b || same(found[k], b);
            }
            if (has_a && has_b) {
                return u;
            }
        }
        return -1;
    }

private:
    const VoronoiVertices& v_;
};

std::array<Corner, 3> relative(const std::array<Corner, 4>& tet, int from) {
    // the other three corners seen from corner from, in order
    std::array<Corner, 3> out;
    int k = 0;
    for (int c = 0; c < 4; ++c) {
        if (c == from) {
            continue;
        }
        Corner corner = tet[c];
        for (int d = 0; d < 3; ++d) {
            corner.shift[d] = static_cast<std::int8_t>(tet[c].shift[d] - tet[from].shift[d]);
        }
        out[k++] = corner;
    }
    std::sort(out.begin(), out.end(), before);
    return out;
}

}  // namespace

Tetrahedra delaunay(const VoronoiVertices& c, const Control& control) {
    const Cells cells(c);
    const auto n = static_cast<std::int64_t>(c.offsets.size()) - 1;
    const std::int64_t vertices = c.offsets[n];
    // a tetrahedron once, from the cell of its lowest atom
    std::vector<std::int64_t> owned(static_cast<std::size_t>(n + 1), 0);
    std::atomic<bool> repeated{false};
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::array<Corner, 4> tet;
            bool seen = false;
            for (std::int64_t i = begin; i < end; ++i) {
                std::int64_t mine = 0;
                for (std::int64_t v = c.offsets[i]; v < c.offsets[i + 1]; ++v) {
                    const int kind = cells.corners(i, v, tet);
                    seen = seen || kind == 2;
                    mine += kind == 0 && i < tet[1].atom ? 1 : 0;
                }
                owned[i + 1] = mine;
            }
            if (seen) {
                repeated = true;
            }
        },
        stage(control, 0, 3));
    for (std::int64_t i = 0; i < n; ++i) {
        owned[i + 1] += owned[i];
    }
    const std::int64_t count = owned[n];
    Tetrahedra out;
    out.repeated = repeated;
    out.atoms.resize(static_cast<std::size_t>(4 * count));
    out.shifts.resize(static_cast<std::size_t>(12 * count));
    out.neighbors.assign(static_cast<std::size_t>(4 * count), -1);
    out.sides.assign(static_cast<std::size_t>(4 * count), -1);
    out.radii.resize(static_cast<std::size_t>(count));
    std::vector<std::int64_t> tet_of(static_cast<std::size_t>(vertices), -1);
    // the tetrahedra owned
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::array<Corner, 4> tet;
            for (std::int64_t i = begin; i < end; ++i) {
                std::int64_t t = owned[i];
                for (std::int64_t v = c.offsets[i]; v < c.offsets[i + 1]; ++v) {
                    if (cells.corners(i, v, tet) != 0 || tet[1].atom < i) {
                        continue;
                    }
                    tet_of[v] = t;
                    out.radii[t] = cells.radius(i, v);
                    for (int k = 0; k < 4; ++k) {
                        out.atoms[4 * t + k] = tet[k].atom;
                        for (int d = 0; d < 3; ++d) {
                            out.shifts[12 * t + 3 * k + d] = tet[k].shift[d];
                        }
                    }
                    ++t;
                }
            }
        },
        stage(control, 1, 3));
    // the others: the same tetrahedron in the cell of its lowest atom
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::array<Corner, 4> tet;
            for (std::int64_t i = begin; i < end; ++i) {
                for (std::int64_t v = c.offsets[i]; v < c.offsets[i + 1]; ++v) {
                    if (cells.corners(i, v, tet) != 0 || tet[1].atom > i) {
                        continue;
                    }
                    const std::int64_t w = cells.find(tet[1].atom, relative(tet, 1));
                    if (w >= 0) {
                        tet_of[v] = tet_of[w];
                    }
                }
            }
        },
        control);
    // the tetrahedra across the faces: the other ends of the edges of the
    // cells, which lie along the faces between their atoms
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::array<Corner, 4> tet, other;
            for (std::int64_t i = begin; i < end; ++i) {
                for (std::int64_t v = c.offsets[i]; v < c.offsets[i + 1]; ++v) {
                    if (cells.corners(i, v, tet) != 0 || tet[1].atom < i) {
                        continue;
                    }
                    const std::int64_t t = tet_of[v];
                    for (int side = 0; side < 4; ++side) {
                        std::int64_t u = -1;
                        if (side > 0) {
                            // in this cell, toward the two other corners
                            const Corner& a = tet[side == 1 ? 2 : 1];
                            const Corner& b = tet[side == 3 ? 2 : 3];
                            u = cells.across(i, v, a, b);
                        } else {
                            // in the cell of corner 1, toward corners 2 and 3
                            const std::int64_t m = tet[1].atom;
                            const std::int64_t w = cells.find(m, relative(tet, 1));
                            if (w < 0 || cells.corners(m, w, other) != 0) {
                                continue;
                            }
                            std::array<const Corner*, 2> toward{};
                            int k2 = 0;
                            for (int k = 1; k < 4; ++k) {
                                if (other[k].atom != i) {
                                    toward[static_cast<std::size_t>(k2++)] = &other[k];
                                }
                            }
                            u = cells.across(m, w, *toward[0], *toward[1]);
                        }
                        if (u < 0 || tet_of[u] < 0) {
                            continue;
                        }
                        const std::int64_t next = tet_of[u];
                        out.neighbors[4 * t + side] = next;
                        // the corner of that one off the shared face
                        for (int k = 0; k < 4; ++k) {
                            const std::int64_t atom = out.atoms[4 * next + k];
                            bool shared = false;
                            for (int q = 0; q < 4; ++q) {
                                shared = shared || (q != side && out.atoms[4 * t + q] == atom);
                            }
                            if (!shared) {
                                out.sides[4 * t + side] = static_cast<std::int8_t>(k);
                            }
                        }
                    }
                }
            }
        },
        stage(control, 2, 3));
    return out;
}

}  // namespace mdapy::analysis

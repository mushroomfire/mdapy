// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/isosurface.h"

#include <cmath>
#include <unordered_map>
#include <utility>

namespace mdapy::analysis {

namespace {

using Vec = std::array<double, 3>;
using Corner = std::array<int, 3>;  // 0 or 1 along each axis of a cell

// the six tetrahedra of a cell: from corner 000 along one axis, then
// another, then to corner 111
constexpr std::array<std::array<int, 3>, 6> kOrders = {
    {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};

Vec minus(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

double triple(const Vec& a, const Vec& b, const Vec& c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
           a[2] * (b[0] * c[1] - b[1] * c[0]);
}

}  // namespace

Isosurface isosurface(const double* values, const std::array<std::int64_t, 3>& n,
                      const bool* pbc, double level, bool left_handed) {
    Isosurface out;
    std::array<std::int64_t, 3> cells{};
    for (std::size_t e = 0; e < 3; ++e) {
        cells[e] = pbc[e] ? n[e] : n[e] - 1;
        if (cells[e] <= 0) {
            return out;
        }
    }
    auto sample = [&](const std::array<std::int64_t, 3>& g) {
        return (g[0] * n[1] + g[1]) * n[2] + g[2];
    };
    // the vertex on the edge from a sample along a direction (1 to 7, its
    // bits the axes), made once for the cells that share the edge
    std::unordered_map<std::int64_t, std::int64_t> made;
    auto vertex = [&](const std::array<std::int64_t, 3>& g, int direction, double t) {
        const std::int64_t key = sample(g) * 8 + direction;
        const auto found = made.find(key);
        if (found != made.end()) {
            return found->second;
        }
        const auto index = static_cast<std::int64_t>(out.vertices.size() / 3);
        for (std::size_t e = 0; e < 3; ++e) {
            double f = (static_cast<double>(g[e]) + 0.5 + t * ((direction >> e) & 1)) /
                       static_cast<double>(n[e]);
            if (pbc[e]) {
                f -= std::floor(f);
            }
            out.vertices.push_back(f);
        }
        made.emplace(key, index);
        return index;
    };
    std::array<std::int64_t, 3> c{};
    for (c[0] = 0; c[0] < cells[0]; ++c[0]) {
        for (c[1] = 0; c[1] < cells[1]; ++c[1]) {
            for (c[2] = 0; c[2] < cells[2]; ++c[2]) {
                for (const auto& order : kOrders) {
                    // the corners of the tetrahedron, each one step further
                    std::array<Corner, 4> corner{};
                    for (int s = 1; s < 4; ++s) {
                        corner[static_cast<std::size_t>(s)] = corner[static_cast<std::size_t>(s - 1)];
                        corner[static_cast<std::size_t>(s)][static_cast<std::size_t>(order[static_cast<std::size_t>(s - 1)])] = 1;
                    }
                    std::array<double, 4> v{};
                    std::array<std::array<std::int64_t, 3>, 4> g{};  // the samples, in the grid
                    std::array<Vec, 4> at{};  // fractional, as this cell places them
                    int inside = 0;
                    for (std::size_t s = 0; s < 4; ++s) {
                        for (std::size_t e = 0; e < 3; ++e) {
                            const std::int64_t u = c[e] + corner[s][e];
                            g[s][e] = u % n[e];
                            at[s][e] = (static_cast<double>(u) + 0.5) / static_cast<double>(n[e]);
                        }
                        v[s] = values[sample(g[s])];
                        if (v[s] > level) {
                            inside |= 1 << s;
                        }
                    }
                    if (inside == 0 || inside == 15) {
                        continue;
                    }
                    // the crossing on the edge between corners s < t: its
                    // vertex and where this cell places it
                    auto cross = [&](std::size_t s, std::size_t t) {
                        int direction = 0;
                        for (std::size_t e = 0; e < 3; ++e) {
                            direction |= (corner[t][e] - corner[s][e]) << e;
                        }
                        const double f = (level - v[s]) / (v[t] - v[s]);
                        Vec p{};
                        for (std::size_t e = 0; e < 3; ++e) {
                            p[e] = at[s][e] + f * (at[t][e] - at[s][e]);
                        }
                        return std::make_pair(vertex(g[s], direction, f), p);
                    };
                    auto edge = [&](std::size_t a, std::size_t b) {
                        return a < b ? cross(a, b) : cross(b, a);
                    };
                    std::vector<std::size_t> in, away;
                    Vec mid_in{}, mid_out{};
                    for (std::size_t s = 0; s < 4; ++s) {
                        auto& group = (inside >> s) & 1 ? in : away;
                        auto& mid = (inside >> s) & 1 ? mid_in : mid_out;
                        group.push_back(s);
                        for (std::size_t e = 0; e < 3; ++e) {
                            mid[e] += at[s][e];
                        }
                    }
                    Vec outward{};
                    for (std::size_t e = 0; e < 3; ++e) {
                        outward[e] = mid_out[e] / static_cast<double>(away.size()) -
                                     mid_in[e] / static_cast<double>(in.size());
                    }
                    std::vector<std::array<std::pair<std::int64_t, Vec>, 3>> triangles;
                    if (in.size() == 1 || away.size() == 1) {
                        const bool lone_in = in.size() == 1;
                        const std::size_t lone = lone_in ? in[0] : away[0];
                        const auto& rest = lone_in ? away : in;
                        triangles.push_back({edge(lone, rest[0]), edge(lone, rest[1]), edge(lone, rest[2])});
                    } else {
                        const auto ac = edge(in[0], away[0]), ad = edge(in[0], away[1]);
                        const auto bd = edge(in[1], away[1]), bc = edge(in[1], away[0]);
                        triangles.push_back({ac, ad, bd});
                        triangles.push_back({ac, bd, bc});
                    }
                    for (auto& triangle : triangles) {
                        const double turn = triple(minus(triangle[1].second, triangle[0].second),
                                                   minus(triangle[2].second, triangle[0].second),
                                                   outward);
                        if ((turn < 0) != left_handed) {
                            std::swap(triangle[1], triangle[2]);
                        }
                        for (const auto& [index, p] : triangle) {
                            out.faces.push_back(index);
                            for (std::size_t e = 0; e < 3; ++e) {
                                const double kept = out.vertices[3 * static_cast<std::size_t>(index) + e];
                                out.shifts.push_back(static_cast<std::int8_t>(std::lround(p[e] - kept)));
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace mdapy::analysis

// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/surface_mesh.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace mdapy::analysis {
namespace {

using Vec = std::array<double, 3>;
using Shift = std::array<std::int32_t, 3>;

// The face opposite each corner of a tetrahedron of positive volume,
// counterclockwise seen from outside.
constexpr int kOutward[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

// Corner k of tetrahedron t with the atoms at x.
Vec corner(const Tetrahedra& tetrahedra, const double* x, const double* matrix, std::int64_t t,
           int k) {
    const std::int64_t atom = tetrahedra.atoms[static_cast<std::size_t>(4 * t + k)];
    const std::int8_t* s = tetrahedra.shifts.data() + 12 * t + 3 * k;
    Vec p;
    for (int d = 0; d < 3; ++d) {
        p[d] = x[3 * atom + d] + s[0] * matrix[d] + s[1] * matrix[3 + d] + s[2] * matrix[6 + d];
    }
    return p;
}

// Six times the volume of a tetrahedron, positive when its corners turn as
// the axes do.
double volume6(const Tetrahedra& tetrahedra, const double* x, const double* matrix,
               std::int64_t t) {
    const Vec p = corner(tetrahedra, x, matrix, t, 0);
    Vec e[3];
    for (int k = 0; k < 3; ++k) {
        const Vec q = corner(tetrahedra, x, matrix, t, k + 1);
        e[k] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
    }
    return e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
           e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
           e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
}

bool faces_surface(const SurfaceMesh& mesh, const Tetrahedra& tetrahedra, std::int64_t t,
                   int side) {
    const std::int64_t next = tetrahedra.neighbors[static_cast<std::size_t>(4 * t + side)];
    return mesh.solid[static_cast<std::size_t>(t)] != 0 &&
           (next < 0 || mesh.solid[static_cast<std::size_t>(next)] == 0);
}

// A neighbor of a vertex: its vertex and the periodic image it is in.
struct Edge {
    std::int64_t to;
    std::array<std::int8_t, 3> shift;
};

}  // namespace

SurfaceMesh alpha_shape(const Tetrahedra& tetrahedra, std::int64_t n, const double* positions,
                        const double* jittered, const double* matrix, double radius,
                        const Control& control) {
    const auto count = static_cast<std::int64_t>(tetrahedra.radii.size());
    SurfaceMesh mesh;
    mesh.solid.resize(static_cast<std::size_t>(count));
    mesh.volumes.resize(static_cast<std::size_t>(count));
    // the turn of each tetrahedron from the atoms it was found from, which
    // is certain also where the atoms lie on a common sphere
    std::vector<std::int8_t> turn(static_cast<std::size_t>(count));
    std::vector<std::int64_t> first(static_cast<std::size_t>(count) + 1, 0);
    parallel_for(
        count, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                const auto at = static_cast<std::size_t>(t);
                mesh.solid[at] = tetrahedra.radii[at] < radius ? 1 : 0;
                turn[at] = volume6(tetrahedra, jittered, matrix, t) < 0.0 ? -1 : 1;
                mesh.volumes[at] = turn[at] * volume6(tetrahedra, positions, matrix, t) / 6.0;
            }
        },
        stage(control, 0, 2));
    parallel_for(
        count, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                std::int64_t sides = 0;
                for (int side = 0; side < 4; ++side) {
                    sides += faces_surface(mesh, tetrahedra, t, side) ? 1 : 0;
                }
                first[static_cast<std::size_t>(t) + 1] = sides;
            }
        },
        control);
    prefix_sum(first.data(), count + 1);
    const std::int64_t triangles = first[static_cast<std::size_t>(count)];
    mesh.faces.resize(static_cast<std::size_t>(3 * triangles));
    mesh.shifts.resize(static_cast<std::size_t>(9 * triangles));
    mesh.cells.resize(static_cast<std::size_t>(2 * triangles));
    parallel_for(
        count, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                std::int64_t f = first[static_cast<std::size_t>(t)];
                for (int side = 0; side < 4; ++side) {
                    if (!faces_surface(mesh, tetrahedra, t, side)) {
                        continue;
                    }
                    int order[3] = {kOutward[side][0], kOutward[side][1], kOutward[side][2]};
                    if (turn[static_cast<std::size_t>(t)] < 0) {
                        std::swap(order[1], order[2]);
                    }
                    const std::int8_t* base = tetrahedra.shifts.data() + 12 * t + 3 * order[0];
                    for (int m = 0; m < 3; ++m) {
                        mesh.faces[static_cast<std::size_t>(3 * f + m)] =
                            tetrahedra.atoms[static_cast<std::size_t>(4 * t + order[m])];
                        const std::int8_t* s = tetrahedra.shifts.data() + 12 * t + 3 * order[m];
                        for (int d = 0; d < 3; ++d) {
                            mesh.shifts[static_cast<std::size_t>(9 * f + 3 * m + d)] =
                                static_cast<std::int8_t>(s[d] - base[d]);
                        }
                    }
                    mesh.cells[static_cast<std::size_t>(2 * f)] = t;
                    mesh.cells[static_cast<std::size_t>(2 * f + 1)] =
                        tetrahedra.neighbors[static_cast<std::size_t>(4 * t + side)];
                    ++f;
                }
            }
        },
        stage(control, 1, 2));
    // the corners of each face in the tetrahedron, as above
    const auto corner_of = [&](std::int64_t face, int m) {
        const std::int64_t t = mesh.cells[static_cast<std::size_t>(2 * face)];
        int side = 0;
        for (int s = 0, seen = 0; s < 4; ++s) {
            if (faces_surface(mesh, tetrahedra, t, s)) {
                if (first[static_cast<std::size_t>(t)] + seen == face) {
                    side = s;
                    break;
                }
                ++seen;
            }
        }
        int order[3] = {kOutward[side][0], kOutward[side][1], kOutward[side][2]};
        if (turn[static_cast<std::size_t>(t)] < 0) {
            std::swap(order[1], order[2]);
        }
        return std::array<int, 4>{order[m], order[(m + 1) % 3], order[(m + 2) % 3], side};
    };
    const auto face_of = [&](std::int64_t t, int side) {
        std::int64_t f = first[static_cast<std::size_t>(t)];
        for (int s = 0; s < side; ++s) {
            f += faces_surface(mesh, tetrahedra, t, s) ? 1 : 0;
        }
        return f;
    };
    const auto atom = [&](std::int64_t t, int k) {
        return tetrahedra.atoms[static_cast<std::size_t>(4 * t + k)];
    };
    // the half-edge of the next face on the same sheet across each edge: the
    // face reached by turning about the edge through the solid tetrahedra
    std::vector<std::int64_t> across(static_cast<std::size_t>(3 * triangles), -1);
    bool closed = true;
    for (std::int64_t f = 0; f < triangles && closed; ++f) {
        for (int m = 0; m < 3; ++m) {
            std::int64_t t = mesh.cells[static_cast<std::size_t>(2 * f)];
            const std::array<int, 4> c = corner_of(f, m);
            const std::int64_t a = atom(t, c[0]), b = atom(t, c[1]);
            int kc = c[2], kd = c[3];
            std::int64_t found = -1;
            for (int step = 0; step < 1000; ++step) {
                const std::int64_t next =
                    tetrahedra.neighbors[static_cast<std::size_t>(4 * t + kc)];
                if (next < 0 || mesh.solid[static_cast<std::size_t>(next)] == 0) {
                    found = face_of(t, kc);
                    break;
                }
                const std::int64_t d = atom(t, kd);
                const int e = tetrahedra.sides[static_cast<std::size_t>(4 * t + kc)];
                int kd2 = e, kc2 = -1;
                for (int k = 0; k < 4; ++k) {
                    if (k != e && atom(next, k) == d) {
                        kc2 = k;
                    }
                }
                if (kc2 < 0) {
                    break;
                }
                t = next;
                kc = kc2;
                kd = kd2;
            }
            std::int64_t partner = -1;
            if (found >= 0) {
                for (int k = 0; k < 3; ++k) {
                    const std::array<int, 4> g = corner_of(found, k);
                    const std::int64_t tg = mesh.cells[static_cast<std::size_t>(2 * found)];
                    if (atom(tg, g[0]) == b && atom(tg, g[1]) == a) {
                        partner = 3 * found + k;
                    }
                }
            }
            if (partner < 0) {
                closed = false;
                break;
            }
            across[static_cast<std::size_t>(3 * f + m)] = partner;
        }
    }
    // a vertex for each fan of faces about an atom, so that two sheets of the
    // surface through one atom smooth apart, as OVITO's; the vertices in the
    // order of their atoms, one for each atom where the surface is a manifold
    std::vector<std::int64_t> start(static_cast<std::size_t>(3 * triangles), -1);
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(n) + 1, 0);
    for (std::int64_t h = 0; h < 3 * triangles; ++h) {
        ++offsets[static_cast<std::size_t>(mesh.faces[static_cast<std::size_t>(h)]) + 1];
    }
    prefix_sum(offsets.data(), n + 1);
    std::vector<std::int64_t> from(static_cast<std::size_t>(3 * triangles));
    {
        std::vector<std::int64_t> fill(offsets.begin(), offsets.end() - 1);
        for (std::int64_t h = 0; h < 3 * triangles; ++h) {
            from[static_cast<std::size_t>(
                fill[static_cast<std::size_t>(mesh.faces[static_cast<std::size_t>(h)])]++)] = h;
        }
    }
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t at = offsets[static_cast<std::size_t>(i)];
             at < offsets[static_cast<std::size_t>(i) + 1]; ++at) {
            const std::int64_t h0 = from[static_cast<std::size_t>(at)];
            if (start[static_cast<std::size_t>(h0)] >= 0) {
                continue;
            }
            const auto v = static_cast<std::int64_t>(mesh.atoms.size());
            mesh.atoms.push_back(i);
            mesh.vertices.insert(mesh.vertices.end(), positions + 3 * i, positions + 3 * i + 3);
            if (!closed) {
                // one vertex for the atom
                for (std::int64_t k = offsets[static_cast<std::size_t>(i)];
                     k < offsets[static_cast<std::size_t>(i) + 1]; ++k) {
                    start[static_cast<std::size_t>(from[static_cast<std::size_t>(k)])] = v;
                }
                break;
            }
            std::int64_t h = h0;
            do {
                start[static_cast<std::size_t>(h)] = v;
                const std::int64_t back = across[static_cast<std::size_t>(h)];
                h = 3 * (back / 3) + (back % 3 + 1) % 3;
            } while (h != h0 && start[static_cast<std::size_t>(h)] < 0);
        }
    }
    for (std::int64_t h = 0; h < 3 * triangles; ++h) {
        mesh.faces[static_cast<std::size_t>(h)] = start[static_cast<std::size_t>(h)];
    }
    return mesh;
}

void smooth_mesh(SurfaceMesh& mesh, const double* matrix, int iterations,
                 const Control& control) {
    const auto vertices = static_cast<std::int64_t>(mesh.atoms.size());
    const auto triangles = static_cast<std::int64_t>(mesh.faces.size() / 3);
    if (iterations <= 0 || triangles == 0) {
        return;
    }
    // the edges out of each vertex, one for each face about it, as OVITO goes
    // round the vertex: a neighbor met twice counts twice
    std::vector<std::int64_t> start(static_cast<std::size_t>(vertices) + 1, 0);
    for (const std::int64_t v : mesh.faces) {
        start[static_cast<std::size_t>(v) + 1] += 1;
    }
    prefix_sum(start.data(), vertices + 1);
    std::vector<Edge> edges(static_cast<std::size_t>(start.back()));
    std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
    for (std::int64_t f = 0; f < triangles; ++f) {
        for (int m = 0; m < 3; ++m) {
            const auto from = static_cast<std::size_t>(3 * f + m);
            const std::int8_t* s = mesh.shifts.data() + 3 * from;
            for (const int other : {(m + 1) % 3}) {
                const auto to = static_cast<std::size_t>(3 * f + other);
                const std::int8_t* r = mesh.shifts.data() + 3 * to;
                Edge edge{mesh.faces[to],
                          {static_cast<std::int8_t>(r[0] - s[0]),
                           static_cast<std::int8_t>(r[1] - s[1]),
                           static_cast<std::int8_t>(r[2] - s[2])}};
                edges[static_cast<std::size_t>(fill[static_cast<std::size_t>(
                    mesh.faces[from])]++)] = edge;
            }
        }
    }
    std::vector<std::int64_t> kept(static_cast<std::size_t>(vertices));
    parallel_for(
        vertices, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t v = begin; v < end; ++v) {
                kept[static_cast<std::size_t>(v)] =
                    start[static_cast<std::size_t>(v) + 1] - start[static_cast<std::size_t>(v)];
            }
        },
        control);
    // Taubin's lambda and mu steps, as OVITO takes them
    const double lambda = 0.5;
    const double mu = 1.0 / (0.1 - 1.0 / lambda);
    std::vector<double> moves(mesh.vertices.size());
    for (int step = 0; step < 2 * iterations; ++step) {
        const double factor = step % 2 == 0 ? lambda : mu;
        parallel_for(
            vertices, 0,
            [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t v = begin; v < end; ++v) {
                    const double* x = mesh.vertices.data() + 3 * v;
                    Vec sum = {0.0, 0.0, 0.0};
                    const std::int64_t from = start[static_cast<std::size_t>(v)];
                    const std::int64_t size = kept[static_cast<std::size_t>(v)];
                    for (std::int64_t k = from; k < from + size; ++k) {
                        const Edge& edge = edges[static_cast<std::size_t>(k)];
                        const double* y = mesh.vertices.data() + 3 * edge.to;
                        for (int d = 0; d < 3; ++d) {
                            sum[d] += y[d] - x[d] + edge.shift[0] * matrix[d] +
                                      edge.shift[1] * matrix[3 + d] +
                                      edge.shift[2] * matrix[6 + d];
                        }
                    }
                    for (int d = 0; d < 3; ++d) {
                        moves[static_cast<std::size_t>(3 * v + d)] =
                            size > 0 ? factor * sum[d] / static_cast<double>(size) : 0.0;
                    }
                }
            },
            stage(control, step, 2 * iterations));
        for (std::size_t k = 0; k < moves.size(); ++k) {
            mesh.vertices[k] += moves[k];
        }
    }
}

Regions surface_regions(const Tetrahedra& tetrahedra, const SurfaceMesh& mesh) {
    const auto count = static_cast<std::int64_t>(tetrahedra.radii.size());
    Regions out;
    out.region.assign(static_cast<std::size_t>(count), -1);
    std::vector<Shift> image(static_cast<std::size_t>(count), Shift{0, 0, 0});
    std::vector<std::int64_t> todo;
    bool outside = false;
    for (std::int64_t t = 0; t < count && !outside; ++t) {
        for (int side = 0; side < 4; ++side) {
            outside = outside || tetrahedra.neighbors[static_cast<std::size_t>(4 * t + side)] < 0;
        }
    }
    auto open = [&](bool solid, bool exterior) {
        out.solid.push_back(solid ? 1 : 0);
        out.exterior.push_back(exterior ? 1 : 0);
        out.periodic.push_back(0);
        out.volumes.push_back(0.0);
        return static_cast<std::int64_t>(out.solid.size()) - 1;
    };
    // spread a region from the tetrahedra in todo, which already belong to
    // it; images are tracked unless it started from several at once
    auto spread = [&](std::int64_t r, bool track) {
        const bool solid = out.solid[static_cast<std::size_t>(r)] != 0;
        while (!todo.empty()) {
            const std::int64_t t = todo.back();
            todo.pop_back();
            out.volumes[static_cast<std::size_t>(r)] += mesh.volumes[static_cast<std::size_t>(t)];
            for (int side = 0; side < 4; ++side) {
                const std::int64_t next =
                    tetrahedra.neighbors[static_cast<std::size_t>(4 * t + side)];
                if (next < 0) {
                    continue;
                }
                if ((mesh.solid[static_cast<std::size_t>(next)] != 0) != solid) {
                    continue;
                }
                // the image of next from a corner both share
                const int shared = side == 0 ? 1 : 0;
                const std::int64_t atom = tetrahedra.atoms[static_cast<std::size_t>(4 * t + shared)];
                int there = 0;
                while (tetrahedra.atoms[static_cast<std::size_t>(4 * next + there)] != atom) {
                    ++there;
                }
                Shift expected;
                for (int d = 0; d < 3; ++d) {
                    expected[static_cast<std::size_t>(d)] =
                        image[static_cast<std::size_t>(t)][static_cast<std::size_t>(d)] +
                        tetrahedra.shifts[static_cast<std::size_t>(12 * t + 3 * shared + d)] -
                        tetrahedra.shifts[static_cast<std::size_t>(12 * next + 3 * there + d)];
                }
                if (out.region[static_cast<std::size_t>(next)] < 0) {
                    out.region[static_cast<std::size_t>(next)] = r;
                    image[static_cast<std::size_t>(next)] = expected;
                    todo.push_back(next);
                } else if (track && image[static_cast<std::size_t>(next)] != expected) {
                    out.periodic[static_cast<std::size_t>(r)] = 1;
                }
            }
        }
    };
    if (outside) {
        const std::int64_t r = open(false, true);
        for (std::int64_t t = 0; t < count; ++t) {
            if (mesh.solid[static_cast<std::size_t>(t)] != 0) {
                continue;
            }
            for (int side = 0; side < 4; ++side) {
                if (tetrahedra.neighbors[static_cast<std::size_t>(4 * t + side)] < 0 &&
                    out.region[static_cast<std::size_t>(t)] < 0) {
                    out.region[static_cast<std::size_t>(t)] = r;
                    todo.push_back(t);
                }
            }
        }
        spread(r, false);
    }
    for (std::int64_t t = 0; t < count; ++t) {
        if (out.region[static_cast<std::size_t>(t)] >= 0) {
            continue;
        }
        const std::int64_t r = open(mesh.solid[static_cast<std::size_t>(t)] != 0, false);
        out.region[static_cast<std::size_t>(t)] = r;
        todo.push_back(t);
        spread(r, true);
    }
    const std::size_t triangles = mesh.cells.size() / 2;
    out.faces.resize(2 * triangles);
    for (std::size_t f = 0; f < triangles; ++f) {
        const std::int64_t solid = mesh.cells[2 * f], empty = mesh.cells[2 * f + 1];
        out.faces[2 * f] = out.region[static_cast<std::size_t>(solid)];
        out.faces[2 * f + 1] = empty < 0 ? 0 : out.region[static_cast<std::size_t>(empty)];
    }
    return out;
}

}  // namespace mdapy::analysis

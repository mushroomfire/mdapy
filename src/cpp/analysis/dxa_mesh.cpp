// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/dxa_mesh.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>

#include "identify/ptm.h"

namespace mdapy::analysis::dxa {
namespace {

// the corners of the face opposite each corner of a positively oriented
// tetrahedron, counterclockwise seen from outside
constexpr int kOutward[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

// the longest walk around an edge through good tetrahedra
constexpr int kAround = 4096;

int corner_of(const Tetrahedra& t, std::int64_t tet, std::int64_t atom) {
    for (int c = 0; c < 4; ++c) {
        if (t.atoms[static_cast<std::size_t>(4 * tet + c)] == atom) {
            return c;
        }
    }
    return -1;
}

}  // namespace

double neighbor_cutoff(const ClusterInput& atoms) {
    const double root2 = std::sqrt(2.0);
    double cutoff = 0.0;
    for (std::int64_t i = 0; i < atoms.n; ++i) {
        const int type = atoms.structure[i];
        int first = 0, last = 0;
        double scale = 0.0;
        if (type == identify::kPtmFcc || type == identify::kPtmHcp) {
            first = 0, last = 12, scale = (1 + root2) / 2;
        } else if (type == identify::kPtmBcc) {
            first = 0, last = 8, scale = (1 + root2) / std::sqrt(3.0);
        } else if (type == identify::kPtmCubicDiamond || type == identify::kPtmHexagonalDiamond) {
            first = 4, last = 16, scale = (1 + root2) / 2;
        } else {
            continue;
        }
        double sum = 0.0;
        int count = 0;
        for (int k = first; k < last; ++k) {
            const std::int64_t j = atoms.neighbors[16 * i + k];
            if (j < 0) {
                continue;
            }
            Vec f{};
            for (int d = 0; d < 3; ++d) {
                const double delta = atoms.positions[3 * j + d] - atoms.positions[3 * i + d];
                for (int e = 0; e < 3; ++e) {
                    f[static_cast<std::size_t>(e)] += delta * atoms.inverse[3 * d + e];
                }
            }
            for (int e = 0; e < 3; ++e) {
                if (atoms.pbc[e]) {
                    f[static_cast<std::size_t>(e)] -= std::round(f[static_cast<std::size_t>(e)]);
                }
            }
            double length = 0.0;
            for (int d = 0; d < 3; ++d) {
                double v = 0.0;
                for (int e = 0; e < 3; ++e) {
                    v += f[static_cast<std::size_t>(e)] * atoms.matrix[3 * e + d];
                }
                length += v * v;
            }
            sum += std::sqrt(length);
            ++count;
        }
        if (count > 0) {
            cutoff = std::max(cutoff, scale * sum / count);
        }
    }
    return cutoff;
}

InterfaceMesh interface_mesh(const ClusterInput& atoms, const Mapping& mapping,
                             const Tetrahedra& tetrahedra, const std::vector<std::uint8_t>& good,
                             const double* jittered, double alpha, const Control& control) {
    const auto count = static_cast<std::int64_t>(tetrahedra.radii.size());
    std::vector<std::uint8_t> inside(static_cast<std::size_t>(count));
    for (std::int64_t t = 0; t < count; ++t) {
        const double r = tetrahedra.radii[static_cast<std::size_t>(t)];
        inside[static_cast<std::size_t>(t)] = good[static_cast<std::size_t>(t)] && r * r < alpha;
    }
    // a corner of a tetrahedron in its periodic image
    auto corner = [&](const double* at, std::int64_t tet, int k) {
        const std::int64_t atom = tetrahedra.atoms[static_cast<std::size_t>(4 * tet + k)];
        const std::int8_t* s = tetrahedra.shifts.data() + 12 * tet + 3 * k;
        Vec p{};
        for (int d = 0; d < 3; ++d) {
            p[static_cast<std::size_t>(d)] = at[3 * atom + d] + s[0] * atoms.matrix[d] +
                                             s[1] * atoms.matrix[3 + d] +
                                             s[2] * atoms.matrix[6 + d];
        }
        return p;
    };
    // the faces of the good tetrahedra toward the rest
    std::vector<std::int64_t> face_of(static_cast<std::size_t>(4 * count), -1);
    std::vector<std::int64_t> face_tet;
    std::vector<std::int8_t> face_side, corners;
    for (std::int64_t t = 0; t < count; ++t) {
        if (!inside[static_cast<std::size_t>(t)]) {
            continue;
        }
        const Vec p0 = corner(jittered, t, 0);
        const double volume =
            dot(minus(corner(jittered, t, 1), p0),
                cross(minus(corner(jittered, t, 2), p0), minus(corner(jittered, t, 3), p0)));
        for (int s = 0; s < 4; ++s) {
            const std::int64_t other = tetrahedra.neighbors[static_cast<std::size_t>(4 * t + s)];
            if (other >= 0 && inside[static_cast<std::size_t>(other)]) {
                continue;
            }
            face_of[static_cast<std::size_t>(4 * t + s)] =
                static_cast<std::int64_t>(face_tet.size());
            face_tet.push_back(t);
            face_side.push_back(static_cast<std::int8_t>(s));
            // counterclockwise seen from inside
            const int* order = kOutward[s];
            corners.push_back(static_cast<std::int8_t>(order[0]));
            corners.push_back(static_cast<std::int8_t>(volume > 0 ? order[2] : order[1]));
            corners.push_back(static_cast<std::int8_t>(volume > 0 ? order[1] : order[2]));
        }
    }
    InterfaceMesh mesh;
    const auto faces = static_cast<std::int64_t>(face_tet.size());
    const std::int64_t n = 3 * faces;
    std::vector<std::int64_t> from_atom(static_cast<std::size_t>(n));
    mesh.physical.resize(static_cast<std::size_t>(3 * n));
    for (std::int64_t e = 0; e < n; ++e) {
        const std::int64_t f = e / 3, tet = face_tet[static_cast<std::size_t>(f)];
        const int u = corners[static_cast<std::size_t>(e)];
        const int v = corners[static_cast<std::size_t>(InterfaceMesh::next(e))];
        from_atom[static_cast<std::size_t>(e)] = tetrahedra.atoms[static_cast<std::size_t>(4 * tet + u)];
        const Vec d = minus(corner(atoms.positions, tet, v), corner(atoms.positions, tet, u));
        for (int k = 0; k < 3; ++k) {
            mesh.physical[static_cast<std::size_t>(3 * e + k)] = d[static_cast<std::size_t>(k)];
            if (!atoms.pbc[k]) {
                continue;
            }
            double f_k = 0.0;
            for (int j = 0; j < 3; ++j) {
                f_k += d[static_cast<std::size_t>(j)] * atoms.inverse[3 * j + k];
            }
            if (std::abs(f_k) >= 0.5) {
                throw std::invalid_argument("an edge of the mesh spans half the box");
            }
        }
    }
    // the ideal vectors, each edge from its atom with the lower index, and
    // the matrices between the frames of the clusters
    const auto& vertex_cluster = mapping.vertex_cluster();
    mesh.ideal.resize(static_cast<std::size_t>(3 * n));
    mesh.cluster.resize(static_cast<std::size_t>(n));
    mesh.back.resize(static_cast<std::size_t>(n));
    mesh.matrices.push_back(kIdentity);
    std::unordered_map<std::int64_t, std::int32_t> matrix_of;
    const std::int64_t clusters = atoms.n;  // more than there are clusters
    for (std::int64_t e = 0; e < n; ++e) {
        const std::int64_t a = from_atom[static_cast<std::size_t>(e)];
        const std::int64_t b = from_atom[static_cast<std::size_t>(InterfaceMesh::next(e))];
        const std::int64_t ca = vertex_cluster[static_cast<std::size_t>(a)];
        const std::int64_t cb = vertex_cluster[static_cast<std::size_t>(b)];
        const auto m = mapping.transition(cb, ca);
        std::optional<Vec> v;
        if (a < b) {
            v = mapping.ideal(a, b);
        } else if (const auto w = mapping.ideal(b, a); w && m) {
            const Vec t = turn(*m, *w);
            v = Vec{-t[0], -t[1], -t[2]};
        }
        if (!v || !m) {
            throw std::logic_error("an edge of a good tetrahedron without an ideal vector");
        }
        for (int k = 0; k < 3; ++k) {
            mesh.ideal[static_cast<std::size_t>(3 * e + k)] = (*v)[static_cast<std::size_t>(k)];
        }
        mesh.cluster[static_cast<std::size_t>(e)] = ca;
        std::int32_t index = 0;
        if (ca != cb) {
            const auto [at, added] = matrix_of.try_emplace(
                cb * clusters + ca, static_cast<std::int32_t>(mesh.matrices.size()));
            if (added) {
                mesh.matrices.push_back(*m);
            }
            index = at->second;
        }
        mesh.back[static_cast<std::size_t>(e)] = index;
    }
    // the opposite half-edge: around the edge through good tetrahedra to the
    // next face of the mesh
    mesh.opposite.assign(static_cast<std::size_t>(n), -1);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t e = begin; e < end; ++e) {
                const std::int64_t f = e / 3;
                std::int64_t tet = face_tet[static_cast<std::size_t>(f)];
                const std::int64_t a = from_atom[static_cast<std::size_t>(e)];
                const std::int64_t b = from_atom[static_cast<std::size_t>(InterfaceMesh::next(e))];
                int through = 6 - corner_of(tetrahedra, tet, a) - corner_of(tetrahedra, tet, b) -
                              face_side[static_cast<std::size_t>(f)];
                for (int step = 0; step < kAround; ++step) {
                    const std::int64_t other =
                        tetrahedra.neighbors[static_cast<std::size_t>(4 * tet + through)];
                    if (other < 0 || !inside[static_cast<std::size_t>(other)]) {
                        const std::int64_t g = face_of[static_cast<std::size_t>(4 * tet + through)];
                        for (int k = 0; k < 3; ++k) {
                            const std::int64_t h = 3 * g + k;
                            if (from_atom[static_cast<std::size_t>(h)] == b &&
                                from_atom[static_cast<std::size_t>(InterfaceMesh::next(h))] == a) {
                                mesh.opposite[static_cast<std::size_t>(e)] = h;
                            }
                        }
                        break;
                    }
                    const int side = tetrahedra.sides[static_cast<std::size_t>(4 * tet + through)];
                    through = 6 - corner_of(tetrahedra, other, a) -
                              corner_of(tetrahedra, other, b) - side;
                    tet = other;
                }
            }
        },
        control);
    for (std::int64_t e = 0; e < n; ++e) {
        const std::int64_t o = mesh.opposite[static_cast<std::size_t>(e)];
        if (o < 0 || mesh.opposite[static_cast<std::size_t>(o)] != e) {
            throw std::logic_error("the interface mesh is not closed");
        }
    }
    // the half-edges from each atom, then a vertex for each disk of
    // triangles around it
    std::vector<std::int64_t> atom_offsets(static_cast<std::size_t>(atoms.n) + 1, 0);
    for (std::int64_t e = 0; e < n; ++e) {
        ++atom_offsets[static_cast<std::size_t>(from_atom[static_cast<std::size_t>(e)]) + 1];
    }
    for (std::int64_t i = 0; i < atoms.n; ++i) {
        atom_offsets[static_cast<std::size_t>(i) + 1] += atom_offsets[static_cast<std::size_t>(i)];
    }
    std::vector<std::int64_t> atom_edges(static_cast<std::size_t>(n));
    {
        std::vector<std::int64_t> fill(atom_offsets.begin(), atom_offsets.end() - 1);
        for (std::int64_t e = 0; e < n; ++e) {
            atom_edges[static_cast<std::size_t>(
                fill[static_cast<std::size_t>(from_atom[static_cast<std::size_t>(e)])]++)] = e;
        }
    }
    mesh.start.assign(static_cast<std::size_t>(n), -1);
    for (std::int64_t i = 0; i < atoms.n; ++i) {
        for (std::int64_t at = atom_offsets[static_cast<std::size_t>(i)];
             at < atom_offsets[static_cast<std::size_t>(i) + 1]; ++at) {
            const std::int64_t first = atom_edges[static_cast<std::size_t>(at)];
            if (mesh.start[static_cast<std::size_t>(first)] >= 0) {
                continue;
            }
            const std::int64_t vertex = mesh.vertices();
            mesh.atoms.push_back(i);
            for (int d = 0; d < 3; ++d) {
                mesh.positions.push_back(atoms.positions[3 * i + d]);
            }
            std::int64_t e = first;
            do {
                mesh.start[static_cast<std::size_t>(e)] = vertex;
                e = InterfaceMesh::next(mesh.opposite[static_cast<std::size_t>(e)]);
            } while (e != first);
        }
    }
    const std::int64_t vertices = mesh.vertices();
    mesh.offsets.assign(static_cast<std::size_t>(vertices) + 1, 0);
    for (std::int64_t e = 0; e < n; ++e) {
        ++mesh.offsets[static_cast<std::size_t>(mesh.start[static_cast<std::size_t>(e)]) + 1];
    }
    for (std::int64_t v = 0; v < vertices; ++v) {
        mesh.offsets[static_cast<std::size_t>(v) + 1] += mesh.offsets[static_cast<std::size_t>(v)];
    }
    mesh.edges.resize(static_cast<std::size_t>(n));
    std::vector<std::int64_t> fill(mesh.offsets.begin(), mesh.offsets.end() - 1);
    for (std::int64_t e = 0; e < n; ++e) {
        mesh.edges[static_cast<std::size_t>(
            fill[static_cast<std::size_t>(mesh.start[static_cast<std::size_t>(e)])]++)] = e;
    }
    return mesh;
}

}  // namespace mdapy::analysis::dxa

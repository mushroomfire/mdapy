// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/dislocations.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "analysis/dxa_mapping.h"
#include "analysis/dxa_mesh.h"
#include "analysis/tessellation.h"
#include "analysis/voronoi.h"
#include "neighbor/cell_list.h"

namespace mdapy::analysis {
namespace {

// Clusters of the crystal's structure smaller than this are slivers, such as
// the one layer of crystal inside an extrinsic stacking fault: their frame
// rests on a few atoms.
constexpr std::int64_t kSliver = 100;

// About a line, the atoms of its cluster closer than this (Å) to a point of
// it fit the lattice there; turned from the frame of the cluster by more
// than kTurned, with at least kFewest atoms, the lattice there gives the
// spatial Burgers vector instead of the cluster's mean orientation, as when
// one cluster holds two grains.
constexpr double kAbout = 7.0;
constexpr double kTurned = 3.0 * 3.14159265358979323846 / 180.0;
constexpr std::int64_t kFewest = 10;

// The rotation of the polar decomposition of m, by Newton's iteration.
dxa::Mat rotation_of(dxa::Mat r) {
    for (int step = 0; step < 100; ++step) {
        const auto inverse = dxa::inverse(r);
        if (!inverse) {
            break;
        }
        const dxa::Mat back = dxa::transpose(*inverse);
        double change = 0.0;
        for (std::size_t k = 0; k < 9; ++k) {
            const double next = 0.5 * (r[k] + back[k]);
            change = std::max(change, std::abs(next - r[k]));
            r[k] = next;
        }
        if (change < 1e-15) {
            break;
        }
    }
    return r;
}

// The vector from atom a to atom b, the nearest image along periodic axes.
dxa::Vec bond(const dxa::ClusterInput& input, std::int64_t a, std::int64_t b) {
    const double* x = input.positions;
    double f[3];
    for (int e = 0; e < 3; ++e) {
        f[e] = 0.0;
        for (int k = 0; k < 3; ++k) {
            f[e] += (x[3 * b + k] - x[3 * a + k]) * input.inverse[3 * k + e];
        }
        if (input.pbc[e]) {
            f[e] -= std::round(f[e]);
        }
    }
    dxa::Vec d{};
    for (int k = 0; k < 3; ++k) {
        d[static_cast<std::size_t>(k)] =
            f[0] * input.matrix[k] + f[1] * input.matrix[3 + k] + f[2] * input.matrix[6 + k];
    }
    return d;
}

// The spatial Burgers vector of each line (3 each): its vector turned by the
// orientation of its cluster, or, where the lattice about the line is turned
// from the cluster's frame, by that rotation times the stretch of the cluster.
std::vector<double> spatial_burgers(const dxa::Network& network, const dxa::Clusters& clusters,
                                    const dxa::ClusterInput& input, const double* origin,
                                    const bool* pbc) {
    const std::int64_t n = input.n;
    const auto& segments = network.segments;
    // the points of the lines, put into the box, after the atoms
    std::vector<double> places(input.positions, input.positions + 3 * n);
    std::vector<std::int64_t> owner;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const std::vector<double>& points = segments[s].points;
        for (std::size_t q = 0; q + 2 < points.size(); q += 3) {
            double f[3];
            for (int e = 0; e < 3; ++e) {
                f[e] = 0.0;
                for (int k = 0; k < 3; ++k) {
                    f[e] += (points[q + static_cast<std::size_t>(k)] - origin[k]) *
                            input.inverse[3 * k + e];
                }
                if (pbc[e]) {
                    f[e] -= std::floor(f[e]);
                }
            }
            for (int k = 0; k < 3; ++k) {
                places.push_back(origin[k] + f[0] * input.matrix[k] + f[1] * input.matrix[3 + k] +
                                 f[2] * input.matrix[6 + k]);
            }
            owner.push_back(static_cast<std::int64_t>(s));
        }
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> near;  // (line, atom)
    if (!owner.empty() && n > 0) {
        const auto total = n + static_cast<std::int64_t>(owner.size());
        const neighbor::CellGrid grid(places.data(), total, input.matrix, origin, pbc, kAbout);
        for (std::int64_t t = 0; t < total; ++t) {
            const auto i = static_cast<std::int64_t>(grid.atom(t));
            if (i < n) {
                continue;
            }
            const std::int64_t s = owner[static_cast<std::size_t>(i - n)];
            const std::int64_t c = segments[static_cast<std::size_t>(s)].cluster;
            grid.visit(t, [&](index_t j, const std::int32_t*, double) {
                if (j < n && c >= 0 && clusters.cluster[static_cast<std::size_t>(j)] == c) {
                    near.emplace_back(s, static_cast<std::int64_t>(j));
                }
            });
        }
        std::sort(near.begin(), near.end());
        near.erase(std::unique(near.begin(), near.end()), near.end());
    }
    // the sums of the fits of the lattice about each line
    std::vector<dxa::Mat> real(segments.size(), dxa::Mat{}), ideal(segments.size(), dxa::Mat{});
    std::vector<std::int64_t> count(segments.size(), 0);
    for (const auto& [s, a] : near) {
        const dxa::Lattice* l = dxa::lattice(input.structure[a]);
        if (l == nullptr) {
            continue;
        }
        const dxa::Mat& turn = l->group[static_cast<std::size_t>(clusters.rotation[static_cast<std::size_t>(a)])];
        dxa::Mat& r = real[static_cast<std::size_t>(s)];
        dxa::Mat& v2 = ideal[static_cast<std::size_t>(s)];
        for (int k = 0; k < l->neighbors; ++k) {
            const auto j = static_cast<std::int64_t>(input.neighbors[16 * a + k]);
            if (j < 0) {
                continue;
            }
            const dxa::Vec v = dxa::turn(turn, l->vectors[static_cast<std::size_t>(k)]);
            const dxa::Vec d = bond(input, a, j);
            for (std::size_t p = 0; p < 3; ++p) {
                for (std::size_t q = 0; q < 3; ++q) {
                    r[3 * p + q] += d[p] * v[q];
                    v2[3 * p + q] += v[p] * v[q];
                }
            }
        }
        ++count[static_cast<std::size_t>(s)];
    }
    std::vector<double> out;
    out.reserve(3 * segments.size());
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const dxa::Segment& segment = segments[s];
        dxa::Mat orientation{};
        if (segment.cluster >= 0) {
            std::copy_n(clusters.orientation.begin() + 9 * segment.cluster, 9, orientation.begin());
        }
        dxa::Vec spatial = dxa::turn(orientation, segment.burgers);
        const auto fitted = dxa::inverse(ideal[s]);
        if (segment.cluster >= 0 && count[s] >= kFewest && fitted) {
            const dxa::Mat here = rotation_of(dxa::multiply(real[s], *fitted));
            const dxa::Mat mean = rotation_of(orientation);
            const dxa::Mat between = dxa::multiply(here, dxa::transpose(mean));
            const double cosine =
                std::clamp(0.5 * (between[0] + between[4] + between[8] - 1.0), -1.0, 1.0);
            if (std::acos(cosine) > kTurned) {
                const dxa::Mat stretch = dxa::multiply(dxa::transpose(mean), orientation);
                spatial = dxa::turn(dxa::multiply(here, stretch), segment.burgers);
            }
        }
        out.insert(out.end(), spatial.begin(), spatial.end());
    }
    return out;
}

// A line traced in a sliver of the crystal gets its Burgers vector in the
// frame of the largest cluster of the crystal related to the sliver, through
// one cluster at most; the frame of a sliver is fitted to a few atoms only.
void out_of_slivers(dxa::Network& network, const dxa::Clusters& clusters,
                    const dxa::Mapping& mapping, int crystal) {
    const auto count = static_cast<std::int64_t>(clusters.structure.size());
    for (dxa::Segment& segment : network.segments) {
        const std::int64_t c = segment.cluster;
        if (c < 0 || clusters.structure[static_cast<std::size_t>(c)] != crystal ||
            clusters.size[static_cast<std::size_t>(c)] >= kSliver) {
            continue;
        }
        std::int64_t best = -1;
        dxa::Mat into{};
        for (std::int64_t k = 0; k < count; ++k) {
            const auto size = clusters.size[static_cast<std::size_t>(k)];
            if (k == c || clusters.structure[static_cast<std::size_t>(k)] != crystal ||
                size <= clusters.size[static_cast<std::size_t>(c)] ||
                (best >= 0 && size <= clusters.size[static_cast<std::size_t>(best)])) {
                continue;
            }
            if (const auto m = mapping.transition(c, k)) {
                best = k;
                into = *m;
            }
        }
        if (best >= 0) {
            segment.burgers = dxa::turn(into, segment.burgers);
            segment.cluster = best;
        }
    }
}

// The mesh of the triangles faces of the interface mesh and of caps, the
// corners of each triangle images of its vertices, the vertices those it
// uses.
SurfaceMesh triangles(const dxa::InterfaceMesh& mesh, const std::vector<std::int64_t>& faces,
                      const std::vector<dxa::Cap>& caps, const double* matrix,
                      const double* inverse, const double* origin, const bool* pbc) {
    SurfaceMesh out;
    std::vector<std::int64_t> index(static_cast<std::size_t>(mesh.vertices()), -1);
    auto vertex = [&](std::int64_t v) {
        auto& at = index[static_cast<std::size_t>(v)];
        if (at < 0) {
            at = static_cast<std::int64_t>(out.atoms.size());
            out.atoms.push_back(mesh.atoms[static_cast<std::size_t>(v)]);
            const double* p = mesh.positions.data() + 3 * v;
            out.vertices.insert(out.vertices.end(), p, p + 3);
        }
        return at;
    };
    // the periodic image of a vertex at the corner at point
    auto corner = [&](std::int64_t v, const dxa::Vec& point) {
        out.faces.push_back(v);
        for (int k = 0; k < 3; ++k) {
            double f = 0.0;
            for (int d = 0; d < 3; ++d) {
                f += (point[static_cast<std::size_t>(d)] -
                      out.vertices[static_cast<std::size_t>(3 * v + d)]) *
                     inverse[3 * d + k];
            }
            out.shifts.push_back(static_cast<std::int8_t>(pbc[k] ? std::lround(f) : 0));
        }
    };
    auto at = [&](std::int64_t v) {
        const double* p = mesh.positions.data() + 3 * v;
        return dxa::Vec{p[0], p[1], p[2]};
    };
    auto physical = [&](std::int64_t e) {
        const double* p = mesh.physical.data() + 3 * e;
        return dxa::Vec{p[0], p[1], p[2]};
    };
    for (const std::int64_t f : faces) {
        dxa::Vec point = at(mesh.start[static_cast<std::size_t>(3 * f)]);
        for (std::int64_t e = 3 * f; e < 3 * f + 3; ++e) {
            corner(vertex(mesh.start[static_cast<std::size_t>(e)]), point);
            point = dxa::add(point, physical(e));
        }
    }
    for (const dxa::Cap& cap : caps) {
        // the end of the line in the box
        dxa::Vec apex = cap.apex;
        for (int k = 0; k < 3; ++k) {
            if (!pbc[k]) {
                continue;
            }
            double f = 0.0;
            for (int d = 0; d < 3; ++d) {
                f += (apex[static_cast<std::size_t>(d)] - origin[d]) * inverse[3 * d + k];
            }
            for (int d = 0; d < 3; ++d) {
                apex[static_cast<std::size_t>(d)] -= std::floor(f) * matrix[3 * k + d];
            }
        }
        const auto tip = static_cast<std::int64_t>(out.atoms.size());
        out.atoms.push_back(-1);
        out.vertices.insert(out.vertices.end(), apex.begin(), apex.end());
        for (const std::int64_t e : cap.edges) {
            const dxa::Vec from = at(mesh.end(e));
            corner(vertex(mesh.end(e)), from);
            corner(vertex(mesh.start[static_cast<std::size_t>(e)]), dxa::minus(from, physical(e)));
            // the image of the end nearest the edge
            dxa::Vec near = dxa::minus(cap.apex, from);
            for (int k = 0; k < 3; ++k) {
                if (!pbc[k]) {
                    continue;
                }
                double f = 0.0;
                for (int d = 0; d < 3; ++d) {
                    f += near[static_cast<std::size_t>(d)] * inverse[3 * d + k];
                }
                for (int d = 0; d < 3; ++d) {
                    near[static_cast<std::size_t>(d)] -= std::round(f) * matrix[3 * k + d];
                }
            }
            corner(tip, dxa::add(from, near));
        }
    }
    return out;
}

// The line whose core holds each atom, -1 for none. The core of a line is
// the bad crystal its circuits enclose: the tetrahedra that are not good and
// not empty space behind the triangles its circuits took for good, and those
// reached from them through such tetrahedra; where they meet those reached
// from the triangles of the other defects, each tetrahedron goes to the
// nearest in steps, a line before the other defects and the first line
// before the others. An atom at a corner of tetrahedra of several lines
// is of the first.
std::vector<std::int64_t> core_atoms(const dxa::InterfaceMesh& mesh, const Tetrahedra& tetrahedra,
                                     const std::vector<std::uint8_t>& good, double alpha,
                                     const std::vector<std::int64_t>& face_line, std::int64_t n) {
    constexpr std::int64_t kFree = -1, kDefect = -2, kNot = -3;
    const std::size_t count = tetrahedra.radii.size();
    // of each tetrahedron: the line, the other defects, not reached yet, or
    // not of the bad crystal
    std::vector<std::int64_t> owner(count);
    for (std::size_t t = 0; t < count; ++t) {
        const double r = tetrahedra.radii[t];
        owner[t] = !good[t] && r * r < alpha ? kFree : kNot;
    }
    auto better = [](std::int64_t a, std::int64_t b) { return a >= 0 && (b < 0 || a < b); };
    std::vector<std::int32_t> depth(count, 0);
    std::vector<std::int64_t> reached;
    for (std::int64_t f = 0; f < mesh.faces(); ++f) {
        const std::int64_t t = mesh.outside[static_cast<std::size_t>(f)];
        if (t < 0 || owner[static_cast<std::size_t>(t)] == kNot) {
            continue;
        }
        const std::int64_t line = face_line[static_cast<std::size_t>(f)];
        const std::int64_t label = line >= 0 ? line : kDefect;
        std::int64_t& of = owner[static_cast<std::size_t>(t)];
        if (of == kFree) {
            reached.push_back(t);
            of = label;
        } else if (better(label, of)) {
            of = label;
        }
    }
    for (std::size_t head = 0; head < reached.size(); ++head) {
        const auto t = static_cast<std::size_t>(reached[head]);
        for (int s = 0; s < 4; ++s) {
            const std::int64_t u = tetrahedra.neighbors[4 * t + static_cast<std::size_t>(s)];
            if (u < 0) {
                continue;
            }
            std::int64_t& of = owner[static_cast<std::size_t>(u)];
            if (of == kFree) {
                of = owner[t];
                depth[static_cast<std::size_t>(u)] = depth[t] + 1;
                reached.push_back(u);
            } else if (of != kNot && depth[static_cast<std::size_t>(u)] == depth[t] + 1 &&
                       better(owner[t], of)) {
                of = owner[t];
            }
        }
    }
    std::vector<std::int64_t> core(static_cast<std::size_t>(n), -1);
    for (const std::int64_t t : reached) {
        const std::int64_t line = owner[static_cast<std::size_t>(t)];
        if (line < 0) {
            continue;
        }
        for (int k = 0; k < 4; ++k) {
            std::int64_t& of = core[static_cast<std::size_t>(
                tetrahedra.atoms[static_cast<std::size_t>(4 * t + k)])];
            if (of < 0 || line < of) {
                of = line;
            }
        }
    }
    return core;
}

}  // namespace

Dislocations dislocation_analysis(const double* positions, const double* jittered, std::int64_t n,
                                  const double* matrix, const double* inverse,
                                  const double* origin, const bool* pbc,
                                  const DislocationOptions& options, const Control& control) {
    if (dxa::lattice(options.crystal) == nullptr) {
        throw std::invalid_argument("no lattice for this structure");
    }
    constexpr int kStages = 5;
    Dislocations out;
    if (options.core) {
        out.core.assign(static_cast<std::size_t>(n), -1);
    }
    out.structures = dxa::find_structures(positions, n, matrix, origin, pbc, options.crystal,
                                          options.defects, stage(control, 0, kStages),
                                          options.low_ca);
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
    // tetrahedra whose circumscribed sphere has a squared radius above five
    // Å times the largest neighbor cutoff are empty space, as in OVITO (the
    // positions are in Å); the walls of the open directions lie beyond the
    // reach of such a sphere
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
    if (options.core) {
        out.core = core_atoms(mesh, tetrahedra, good, alpha, out.network.face_line, n);
    }
    out_of_slivers(out.network, out.clusters, mapping, options.crystal);
    dxa::smooth_lines(out.network, options.separation, options.smoothing);
    out.spatial = spatial_burgers(out.network, out.clusters, input, origin, pbc);
    out.defects =
        triangles(mesh, out.network.faces, out.network.caps, matrix, inverse, origin, pbc);
    smooth_mesh(out.defects, matrix, options.mesh_smoothing);
    if (options.interface) {
        std::vector<std::int64_t> all(static_cast<std::size_t>(mesh.faces()));
        for (std::int64_t f = 0; f < mesh.faces(); ++f) {
            all[static_cast<std::size_t>(f)] = f;
        }
        out.interface = triangles(mesh, all, {}, matrix, inverse, origin, pbc);
    }
    return out;
}

}  // namespace mdapy::analysis

// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "identify/ptm_data.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace mdapy::identify::ptm {
namespace {

using Matrix = std::array<double, 9>;  // row-major

const double kPi = std::acos(-1.0);
const double kSqrt2 = std::sqrt(2.0);
const double kSqrt3 = std::sqrt(3.0);
const double kSqrt6 = std::sqrt(6.0);

Point rotated(const Matrix& m, const Point& p) {
    return {m[0] * p[0] + m[1] * p[1] + m[2] * p[2], m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
            m[6] * p[0] + m[7] * p[1] + m[8] * p[2]};
}

// The rotation by angle (degrees) about an axis.
Matrix turn(Point axis, double degrees) {
    const double length = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    for (double& x : axis) {
        x /= length;
    }
    const double t = degrees * kPi / 180.0, c = std::cos(t), s = std::sin(t);
    const double x = axis[0], y = axis[1], z = axis[2];
    return {c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s,
            y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s,
            z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)};
}

double distance(const Point& a, const Point& b) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) +
                     (a[2] - b[2]) * (a[2] - b[2]));
}

// The index of p among points, or -1.
int find(const std::vector<Point>& points, const Point& p) {
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (distance(points[i], p) < 1e-9) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// ------------------------------------------------------------ templates
//
// The neighbors of each structure in the order OVITO numbers them, at any
// scale; the central atom is added first and the mean distance of the
// neighbors from it made 1.

std::vector<Point> fcc_neighbors() {
    return {{1, 1, 0},  {0, 1, 1},  {1, 0, 1},  {-1, -1, 0}, {0, -1, -1}, {-1, 0, -1},
            {-1, 1, 0}, {0, -1, 1}, {-1, 0, 1}, {1, -1, 0},  {0, 1, -1},  {1, 0, -1}};
}

std::vector<Point> hcp_neighbors() {
    const double h = kSqrt3 / 2, t = kSqrt3 / 6, u = kSqrt3 / 3, z = kSqrt6 / 3;
    return {{0.5, -h, 0}, {-1, 0, 0}, {-0.5, t, -z},  {0.5, t, -z},  {0, -u, -z}, {-0.5, h, 0},
            {0.5, h, 0},  {1, 0, 0},  {-0.5, -h, 0}, {0, -u, z},     {0.5, t, z}, {-0.5, t, z}};
}

std::vector<Point> bcc_neighbors() {
    return {{1, 1, 1},  {-1, 1, 1}, {1, 1, -1},  {-1, -1, 1}, {1, -1, 1},
            {-1, 1, -1}, {-1, -1, -1}, {1, -1, -1}, {2, 0, 0},   {-2, 0, 0},
            {0, 2, 0},  {0, -2, 0}, {0, 0, 2},   {0, 0, -2}};
}

std::vector<Point> ico_neighbors() {
    // two poles, and ten points at a height of +-1/sqrt(5) at angles (degrees)
    const double z = 1 / std::sqrt(5.0), r = 2 / std::sqrt(5.0);
    std::vector<Point> points = {{0, 0, 1}, {0, 0, -1}};
    const double ring[10][2] = {{126, -1}, {-54, 1}, {-90, -1}, {90, 1},   {-18, -1},
                                {162, 1},  {198, -1}, {18, 1},  {54, -1},  {-126, 1}};
    for (const auto& p : ring) {
        const double a = p[0] * kPi / 180.0;
        points.push_back({r * std::cos(a), r * std::sin(a), p[1] * z});
    }
    return points;
}

std::vector<Point> sc_neighbors() {
    return {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
}

// Diamond: the four nearest neighbors, then the three further neighbors
// of each.
std::vector<Point> cubic_diamond_neighbors() {
    return {{1, 1, 1},   {1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {2, 2, 0},  {0, 2, 2},
            {2, 0, 2},   {0, -2, -2}, {2, -2, 0},  {2, 0, -2},  {-2, -2, 0}, {0, -2, 2},
            {-2, 0, 2},  {-2, 0, -2}, {-2, 2, 0},  {0, 2, -2}};
}

std::vector<Point> hexagonal_diamond_neighbors() {
    const double u = 4 * kSqrt2, v = 4 * kSqrt6 / 3, w = 4 * kSqrt3 / 3;
    return {{-u, v, -w},     {0, -2 * v, -w},     {u, v, -w},     {0, 0, 3 * w},
            {-2 * u, 0, 0},  {-u, v, -4 * w},     {-u, 3 * v, 0}, {u, -3 * v, 0},
            {0, -2 * v, -4 * w}, {-u, -3 * v, 0}, {u, v, -4 * w}, {u, 3 * v, 0},
            {2 * u, 0, 0},   {0, -2 * v, 4 * w},  {u, v, 4 * w},  {-u, v, 4 * w}};
}

// Graphene: the three nearest neighbors, then the two further neighbors of
// each, in the plane z = 0.
std::vector<Point> graphene_neighbors() {
    const double h = kSqrt3 / 2;
    return {{0, 1, 0},     {h, -0.5, 0}, {-h, -0.5, 0}, {-h, 1.5, 0}, {h, 1.5, 0},
            {kSqrt3, 0, 0}, {h, -1.5, 0}, {-h, -1.5, 0}, {-kSqrt3, 0, 0}};
}

std::array<Point, kMaxPoints> normalized(const std::vector<Point>& neighbors) {
    double mean = 0.0;
    for (const Point& p : neighbors) {
        mean += distance(p, {0, 0, 0});
    }
    mean /= static_cast<double>(neighbors.size());
    std::array<Point, kMaxPoints> points{};
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        for (int e = 0; e < 3; ++e) {
            points[i + 1][e] = neighbors[i][e] / mean;
        }
    }
    return points;
}

// The pseudo-inverse of a template, row by point: F = sum_k x_k p_k^T fits
// the points x_k best.
std::array<Point, kMaxPoints> pseudo_inverse(const std::array<Point, kMaxPoints>& t, int n) {
    double m[9] = {};
    for (int k = 0; k < n; ++k) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                m[3 * a + b] += t[k][a] * t[k][b];
            }
        }
    }
    double values[3], vectors[9];
    symmetric_eigen(3, m, values, vectors);
    const double largest = *std::max_element(values, values + 3);
    double inverse[9] = {};
    for (int e = 0; e < 3; ++e) {
        if (values[e] > 1e-12 * largest) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    inverse[3 * a + b] += vectors[3 * a + e] * vectors[3 * b + e] / values[e];
                }
            }
        }
    }
    std::array<Point, kMaxPoints> p{};
    for (int k = 0; k < n; ++k) {
        for (int b = 0; b < 3; ++b) {
            for (int a = 0; a < 3; ++a) {
                p[k][b] += t[k][a] * inverse[3 * a + b];
            }
        }
    }
    return p;
}

// ------------------------------------------------------------- symmetry

// The proper rotations that map the points onto themselves.
std::vector<Matrix> rotations_of(const std::vector<Point>& points) {
    // two points not on one line, and every pair the rotations may take
    // them to
    const Point& p = points[0];
    std::size_t second = 1;
    auto across = [](const Point& a, const Point& b) {
        const Point c = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]};
        return distance(c, {0, 0, 0});
    };
    while (across(p, points[second]) < 1e-6) {
        ++second;
    }
    const Point& r = points[second];
    auto frame = [&](const Point& a, const Point& b) {
        Point e1 = a, e2 = b;
        const double na = distance(a, {0, 0, 0});
        for (double& x : e1) {
            x /= na;
        }
        const double along = e1[0] * b[0] + e1[1] * b[1] + e1[2] * b[2];
        for (int k = 0; k < 3; ++k) {
            e2[k] -= along * e1[k];
        }
        const double n2 = distance(e2, {0, 0, 0});
        for (double& x : e2) {
            x /= n2;
        }
        const Point e3 = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                          e1[0] * e2[1] - e1[1] * e2[0]};
        return Matrix{e1[0], e2[0], e3[0], e1[1], e2[1], e3[1], e1[2], e2[2], e3[2]};
    };
    const Matrix from = frame(p, r);
    const double lp = distance(p, {0, 0, 0}), lr = distance(r, {0, 0, 0});
    const double angle = (p[0] * r[0] + p[1] * r[1] + p[2] * r[2]) / (lp * lr);
    std::vector<Matrix> found;
    std::set<std::vector<int>> seen;
    for (const Point& a : points) {
        for (const Point& b : points) {
            const double la = distance(a, {0, 0, 0}), lb = distance(b, {0, 0, 0});
            if (std::fabs(la - lp) > 1e-9 || std::fabs(lb - lr) > 1e-9 ||
                std::fabs((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb) - angle) > 1e-9) {
                continue;
            }
            const Matrix to = frame(a, b);
            Matrix m{};  // to . from^T
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        m[3 * i + j] += to[3 * i + k] * from[3 * j + k];
                    }
                }
            }
            std::vector<int> image;
            for (const Point& q : points) {
                const int k = find(points, rotated(m, q));
                if (k < 0) {
                    break;
                }
                image.push_back(k);
            }
            if (image.size() == points.size() && seen.insert(image).second) {
                found.push_back(m);
            }
        }
    }
    return found;
}

// The quaternions of rotations, each with its first nonzero component
// positive, in decreasing order of w, then x, y and z.
std::vector<Quaternion> group_of(const std::vector<Matrix>& rotations) {
    std::vector<Quaternion> group;
    for (const Matrix& m : rotations) {
        Quaternion q = quaternion(m.data());
        for (double& x : q) {
            if (std::fabs(x) < 1e-12) {
                x = 0.0;
            }
        }
        for (const double x : q) {
            if (x != 0.0) {
                if (x < 0.0) {
                    for (double& y : q) {
                        y = -y;
                    }
                }
                break;
            }
        }
        group.push_back(q);
    }
    std::sort(group.begin(), group.end(), [](const Quaternion& a, const Quaternion& b) {
        for (int e = 0; e < 4; ++e) {
            if (std::fabs(a[e] - b[e]) > 1e-9) {
                return a[e] > b[e];
            }
        }
        return false;
    });
    return group;
}

const std::vector<Quaternion>& cubic_group() {
    static const std::vector<Quaternion> group = group_of(rotations_of(fcc_neighbors()));
    return group;
}

const std::vector<Quaternion>& hexagonal_group() {
    static const std::vector<Quaternion> group = [] {
        std::vector<Point> prism;
        for (int k = 0; k < 6; ++k) {
            const double a = k * kPi / 3.0;
            prism.push_back({std::cos(a), std::sin(a), 1.0});
            prism.push_back({std::cos(a), std::sin(a), -1.0});
        }
        return group_of(rotations_of(prism));
    }();
    return group;
}

const std::vector<Quaternion>& icosahedral_group() {
    static const std::vector<Quaternion> group = group_of(rotations_of(ico_neighbors()));
    return group;
}

// ---------------------------------------------------------------- graphs

std::vector<Facet> normalized_facets(std::vector<Facet> facets) {
    for (Facet& f : facets) {
        const int first = f[0] < f[1] ? (f[0] < f[2] ? 0 : 2) : (f[1] < f[2] ? 1 : 2);
        f = {f[first], f[(first + 1) % 3], f[(first + 2) % 3]};
    }
    std::sort(facets.begin(), facets.end());
    return facets;
}

// The facets of a triangulation of the hull of a template, its quadrilateral
// faces split along either diagonal: one per bit of mask.
std::vector<std::vector<Facet>> triangulations(const std::array<Point, kMaxPoints>& points,
                                              int n) {
    Hull hull;
    Facet facets[kMaxFacets];
    if (grow_hull(hull, n + 1, points.data(), facets) != 0) {
        throw std::logic_error("the hull of a PTM template failed");
    }
    std::vector<Facet> base(facets, facets + hull.count);
    // the diagonals: edges between two facets in one plane
    std::vector<std::pair<int, int>> diagonals;
    for (int f = 0; f < hull.count; ++f) {
        for (int g = f + 1; g < hull.count; ++g) {
            int shared = 0;
            for (const std::int8_t a : base[f]) {
                for (const std::int8_t b : base[g]) {
                    shared += a == b;
                }
            }
            const Point& u = hull.normals[f];
            const Point& v = hull.normals[g];
            if (shared == 2 && u[0] * v[0] + u[1] * v[1] + u[2] * v[2] > 1 - 1e-9) {
                diagonals.emplace_back(f, g);
            }
        }
    }
    std::vector<std::vector<Facet>> all;
    for (std::uint32_t mask = 0; mask < (1U << diagonals.size()); ++mask) {
        std::vector<Facet> t = base;
        for (std::size_t d = 0; d < diagonals.size(); ++d) {
            if (!(mask & (1U << d))) {
                continue;
            }
            // the facets (a, b, c) and (b, a, x) become (b, c, x) and (c, a, x)
            Facet f = t[diagonals[d].first], g = t[diagonals[d].second];
            int a = -1, b = -1, c = -1, x = -1;
            for (int e = 0; e < 3 && a < 0; ++e) {
                for (int k = 0; k < 3; ++k) {
                    if (f[e] == g[(k + 1) % 3] && f[(e + 1) % 3] == g[k]) {
                        a = f[e];
                        b = f[(e + 1) % 3];
                        c = f[(e + 2) % 3];
                        x = g[(k + 2) % 3];
                        break;
                    }
                }
            }
            t[diagonals[d].first] = {static_cast<std::int8_t>(b), static_cast<std::int8_t>(c),
                                     static_cast<std::int8_t>(x)};
            t[diagonals[d].second] = {static_cast<std::int8_t>(c), static_cast<std::int8_t>(a),
                                      static_cast<std::int8_t>(x)};
        }
        all.push_back(normalized_facets(t));
    }
    return all;
}

// Diamond: a facet of three further neighbors of one nearest neighbor
// becomes three facets with it.
std::vector<Facet> with_tents(const std::vector<Facet>& facets) {
    std::vector<Facet> out;
    for (const Facet& f : facets) {
        const int a = f[0], b = f[1], c = f[2];
        if (a >= 4 && b >= 4 && c >= 4 && (a - 4) / 3 == (b - 4) / 3 && (a - 4) / 3 == (c - 4) / 3) {
            const auto inner = static_cast<std::int8_t>((a - 4) / 3);
            out.push_back({inner, f[1], f[2]});
            out.push_back({f[0], inner, f[2]});
            out.push_back({f[0], f[1], inner});
        } else {
            out.push_back(f);
        }
    }
    return normalized_facets(out);
}

// The graphs of a template: one triangulation of each set that its
// rotations take into each other.
std::vector<Graph> graphs_of(const Reference& ref, const std::int8_t* colours, bool tents,
                             int& facets, int& max_degree) {
    const int n = ref.neighbors;
    std::vector<Graph> graphs;
    std::set<std::vector<Facet>> seen;
    facets = 0;
    max_degree = 0;
    for (std::vector<Facet> t : triangulations(ref.templates[0], n)) {
        if (tents) {
            t = with_tents(t);
        }
        if (seen.count(t)) {
            continue;
        }
        for (const Permutation& s : ref.symmetries) {
            std::vector<Facet> image;
            for (const Facet& f : t) {
                image.push_back({static_cast<std::int8_t>(s[f[0] + 1] - 1),
                                 static_cast<std::int8_t>(s[f[1] + 1] - 1),
                                 static_cast<std::int8_t>(s[f[2] + 1] - 1)});
            }
            seen.insert(normalized_facets(image));
        }
        std::int8_t degree[kMaxNeighbors];
        facets = static_cast<int>(t.size());
        max_degree = std::max(max_degree, degrees(facets, t.data(), n, degree));
        Graph graph;
        const auto all = canonical_labellings(facets, t.data(), n, colours, graph.form);
        if (all.empty()) {
            throw std::logic_error("a PTM template graph has no canonical form");
        }
        // a rotation of the template that keeps the triangulation gives an
        // equivalent correspondence: one labelling of each set it relates
        std::vector<const Permutation*> keeping;
        for (const Permutation& s : ref.symmetries) {
            std::vector<Facet> image;
            for (const Facet& f : t) {
                image.push_back({static_cast<std::int8_t>(s[f[0] + 1] - 1),
                                 static_cast<std::int8_t>(s[f[1] + 1] - 1),
                                 static_cast<std::int8_t>(s[f[2] + 1] - 1)});
            }
            if (normalized_facets(image) == t) {
                keeping.push_back(&s);
            }
        }
        std::set<std::array<std::int8_t, kMaxNeighbors>> covered;
        for (const auto& labels : all) {
            if (covered.count(labels)) {
                continue;
            }
            graph.labellings.push_back(labels);
            for (const Permutation* s : keeping) {
                std::array<std::int8_t, kMaxNeighbors> turned{};
                for (int v = 0; v < n; ++v) {
                    turned[v] = labels[(*s)[v + 1] - 1];
                }
                covered.insert(turned);
            }
        }
        graphs.push_back(std::move(graph));
    }
    return graphs;
}

// ------------------------------------------------------------ structures

Reference build(int type) {
    Reference ref;
    ref.type = type;
    std::vector<Point> neighbors;
    std::vector<Matrix> variants = {turn({0, 0, 1}, 0.0)};
    switch (type) {
        case kFcc:
            neighbors = fcc_neighbors();
            ref.group = &cubic_group();
            break;
        case kHcp:
            neighbors = hcp_neighbors();
            ref.group = &hexagonal_group();
            variants.push_back(turn({0, 0, 1}, 60.0));
            break;
        case kBcc:
            neighbors = bcc_neighbors();
            ref.group = &cubic_group();
            break;
        case kIco:
            neighbors = ico_neighbors();
            ref.group = &icosahedral_group();
            break;
        case kSc:
            neighbors = sc_neighbors();
            ref.group = &cubic_group();
            break;
        case kCubicDiamond:
            neighbors = cubic_diamond_neighbors();
            ref.group = &cubic_group();
            variants.push_back(turn({1, 0, 0}, 90.0));
            break;
        case kHexagonalDiamond:
            neighbors = hexagonal_diamond_neighbors();
            ref.group = &hexagonal_group();
            variants.push_back(turn({0, 0, 1}, 60.0));
            variants.push_back(turn({kSqrt3 / 2, 0.5, 0}, 180.0));
            variants.push_back(turn({0.5, kSqrt3 / 2, 0}, 180.0));
            break;
        case kGraphene:
            neighbors = graphene_neighbors();
            ref.group = &hexagonal_group();
            variants.push_back(turn({0, 0, 1}, 60.0));
            break;
        default:
            throw std::logic_error("no PTM structure of this type");
    }
    const int n = static_cast<int>(neighbors.size());
    ref.neighbors = n;
    const std::array<Point, kMaxPoints> first = normalized(neighbors);
    for (const Matrix& m : variants) {
        std::array<Point, kMaxPoints> t{};
        for (int i = 1; i <= n; ++i) {
            t[i] = rotated(m, first[i]);
        }
        ref.templates.push_back(t);
        ref.penrose.push_back(pseudo_inverse(t, n + 1));
    }
    ref.distance = distance(first[1], {0, 0, 0});
    const std::vector<Point> points(first.begin(), first.begin() + n + 1);
    for (const Matrix& m : rotations_of(std::vector<Point>(points.begin() + 1, points.end()))) {
        Permutation s{};
        for (int i = 0; i <= n; ++i) {
            s[i] = static_cast<std::int8_t>(find(points, rotated(m, points[i])));
        }
        ref.symmetries.push_back(s);
    }
    // the conventional group: the template that each element turns the
    // first into, and where its neighbors go
    for (const Quaternion& g : *ref.group) {
        Matrix m;
        rotation_matrix(g, m.data());
        int chosen = -1;
        Permutation moves{};
        for (std::size_t k = 0; k < ref.templates.size() && chosen < 0; ++k) {
            bool all = true;
            for (int j = 0; j <= n && all; ++j) {
                const int i = find(points, rotated(m, ref.templates[k][j]));
                all = i >= 0;
                moves[j] = static_cast<std::int8_t>(i);
            }
            if (all) {
                chosen = static_cast<int>(k);
            }
        }
        if (chosen < 0) {
            throw std::logic_error("an element of the conventional group fits no PTM template");
        }
        ref.group_template.push_back(chosen);
        ref.moves.push_back(moves);
    }
    if (type != kGraphene) {
        std::int8_t colours[kMaxNeighbors] = {};
        const bool diamond = type == kCubicDiamond || type == kHexagonalDiamond;
        if (diamond) {
            std::fill(colours, colours + 4, std::int8_t{1});
        }
        ref.graphs = graphs_of(ref, colours, diamond, ref.facets, ref.max_degree);
    }
    return ref;
}

}  // namespace

const Reference& reference(int type) {
    static std::once_flag once;
    static std::array<Reference, 9> all;
    std::call_once(once, [] {
        for (int t = kFcc; t <= kGraphene; ++t) {
            all[static_cast<std::size_t>(t)] = build(t);
        }
    });
    return all[static_cast<std::size_t>(type)];
}

std::uint32_t canonical_pattern(const Reference& ref, const std::int8_t* other) {
    std::uint32_t best = 0xFFFFFFFFU;
    for (const Permutation& s : ref.symmetries) {
        std::uint32_t code = 0;
        for (int i = 0; i <= ref.neighbors; ++i) {
            code |= static_cast<std::uint32_t>(other[i] != 0) << s[i];
        }
        best = std::min(best, code);
    }
    return best;
}

const FccOrderings& fcc_orderings() {
    static const FccOrderings codes = [] {
        // a neighbor at the centre + d / 2 of a cubic cell of side 1
        const Reference& fcc = reference(kFcc);
        const std::vector<Point> d = fcc_neighbors();
        auto pattern = [&](const Point& centre, auto differs) {
            std::int8_t other[kMaxPoints] = {};
            for (int i = 0; i < fcc.neighbors; ++i) {
                const Point p = {centre[0] + d[i][0] / 2, centre[1] + d[i][1] / 2,
                                 centre[2] + d[i][2] / 2};
                other[i + 1] = static_cast<std::int8_t>(differs(p));
            }
            return canonical_pattern(fcc, other);
        };
        auto whole = [](double x) { return std::fabs(x - std::round(x)) < 1e-9; };
        FccOrderings c{};
        // L1_0, CuAu: Cu on the layers z = 0, Au on z = 1/2
        c.l10 = pattern({0, 0, 0}, [&](const Point& p) { return !whole(p[2]); });
        // L1_2, Cu3Au: Au on the corners, Cu on the faces of the cube
        auto corner = [&](const Point& p) { return whole(p[0]) && whole(p[1]) && whole(p[2]); };
        c.l12_majority = pattern({0.5, 0.5, 0}, corner);
        c.l12_minority = pattern({0, 0, 0}, [&](const Point& p) { return !corner(p); });
        return c;
    }();
    return codes;
}

int into_fundamental_zone(const std::vector<Quaternion>& group, Quaternion& q) {
    double nearest = 0.0;
    int best = -1;
    for (std::size_t i = 0; i < group.size(); ++i) {
        const Quaternion& g = group[i];
        const double t = std::fabs(q[0] * g[0] - q[1] * g[1] - q[2] * g[2] - q[3] * g[3]);
        if (t > nearest) {
            nearest = t;
            best = static_cast<int>(i);
        }
    }
    if (best < 0) {
        return best;
    }
    q = product(q, group[static_cast<std::size_t>(best)]);
    if (q[0] < 0.0) {
        q = {-q[0], -q[1], -q[2], -q[3]};
    }
    return best;
}

}  // namespace mdapy::identify::ptm

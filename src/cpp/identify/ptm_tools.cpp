// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "identify/ptm_tools.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <utility>

namespace mdapy::identify::ptm {
namespace {

// Points closer than this to a plane lie in it; the points are scaled to a
// mean distance of 1 from their centre.
constexpr double kFlat = 1e-12;

double dot(const Point& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point minus(const Point& a, const Point& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Point cross(const Point& a, const Point& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// How far w lies on the side of the plane through p that normal points
// away from: normal . (p - w).
double below(const Point& w, const Point& p, const Point& normal) {
    return dot(normal, minus(p, w));
}

Point unit_normal(const Point* points, int a, int b, int c) {
    Point n = cross(minus(points[b], points[a]), minus(points[c], points[a]));
    const double length = std::sqrt(dot(n, n));
    return {n[0] / length, n[1] / length, n[2] / length};
}

// The first tetrahedron: the two points farthest apart along an axis, the
// point farthest from their line and the one farthest from their plane.
int first_tetrahedron(int n, const Point* points, int* corners) {
    int low[3], high[3];
    for (int axis = 0; axis < 3; ++axis) {
        double smallest = DBL_MAX, largest = -DBL_MAX;
        low[axis] = high[axis] = 0;
        for (int i = 0; i < n; ++i) {
            const double x = points[i][axis];
            if (x < smallest) {
                smallest = x;
                low[axis] = i;
            }
            if (x > largest) {
                largest = x;
                high[axis] = i;
            }
        }
        if (low[axis] == high[axis]) {
            return -1;
        }
    }
    int axis = 0;
    double longest = 0.0;
    for (int e = 0; e < 3; ++e) {
        const Point d = minus(points[low[e]], points[high[e]]);
        if (dot(d, d) > longest) {
            longest = dot(d, d);
            axis = e;
        }
    }
    const int a = low[axis], b = high[axis];
    const Point line = minus(points[b], points[a]);
    const double line2 = dot(line, line);
    int c = -1;
    double farthest = 0.0;
    for (int i = 0; i < n; ++i) {
        if (i == a || i == b) {
            continue;
        }
        const Point d = minus(points[a], points[i]);
        const double along = dot(d, line);
        const double off = (dot(d, d) * line2 - along * along) / line2;
        if (off > farthest) {
            farthest = off;
            c = i;
        }
    }
    if (!(farthest > kFlat)) {
        return -2;
    }
    const Point normal = unit_normal(points, a, b, c);
    int d = -1;
    farthest = 0.0;
    for (int i = 0; i < n; ++i) {
        if (i == a || i == b || i == c) {
            continue;
        }
        const double off = std::fabs(below(points[i], points[a], normal));
        if (off > farthest) {
            farthest = off;
            d = i;
        }
    }
    if (!(farthest > kFlat)) {
        return -3;
    }
    corners[0] = a;
    corners[1] = b;
    corners[2] = c;
    corners[3] = d;
    return 0;
}

// The next node of the facet with the directed edge a -> b, or -1.
struct FacetMap {
    std::int8_t next[kMaxNeighbors][kMaxNeighbors];

    bool build(int count, const Facet* facets) {
        std::memset(next, -1, sizeof(next));
        for (int f = 0; f < count; ++f) {
            const int a = facets[f][0], b = facets[f][1], c = facets[f][2];
            if (next[a][b] != -1 || next[b][c] != -1 || next[c][a] != -1) {
                return false;
            }
            next[a][b] = static_cast<std::int8_t>(c);
            next[b][c] = static_cast<std::int8_t>(a);
            next[c][a] = static_cast<std::int8_t>(b);
        }
        return true;
    }
};

// The walk of Weinberg from the directed edge a -> b: a new node gets the
// next label and the walk turns to the next edge of the facet on its right;
// an old node reached on a new path sends it back; an old node reached on
// an old path sends it out on the first edge to the right not yet walked
// that way. Each label is colour x n + the order of discovery. Compared with
// best as it goes: returns -1 as soon as it is larger, 0 if equal, 1 if
// smaller, when best becomes its code; -2 for an open surface.
int walk(const FacetMap& map, int n, int length, const std::int8_t* colours, int a, int b,
         std::int8_t* best, std::int8_t* order) {
    bool walked[kMaxNeighbors][kMaxNeighbors] = {};
    std::int8_t label[kMaxNeighbors];
    std::memset(label, -1, sizeof(label));
    int found = 0;
    int state = 0;
    auto step = [&](int it, std::int8_t value) {
        if (state == 0) {
            if (value > best[it]) {
                return false;
            }
            if (value < best[it]) {
                state = 1;
            }
        }
        if (state == 1) {
            best[it] = value;
        }
        return true;
    };
    label[a] = static_cast<std::int8_t>(colours[a] * n + found++);
    if (!step(0, label[a])) {
        return -1;
    }
    for (int it = 1; it < length; ++it) {
        const bool fresh = label[b] == -1;
        if (fresh) {
            label[b] = static_cast<std::int8_t>(colours[b] * n + found++);
        }
        if (!step(it, label[b])) {
            return -1;
        }
        int c;
        if (fresh) {
            c = map.next[a][b];
        } else if (!walked[b][a]) {
            c = a;
        } else {
            c = map.next[a][b];
            while (c >= 0 && walked[b][c]) {
                c = map.next[c][b];
            }
        }
        if (c < 0) {
            return -2;
        }
        walked[a][b] = true;
        a = b;
        b = c;
    }
    if (order != nullptr) {
        for (int v = 0; v < n; ++v) {
            order[v] = static_cast<std::int8_t>(label[v] % n);
        }
    }
    return state;
}

std::uint64_t hash(const std::int8_t* code, int length) {
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a
    for (int i = 0; i < length; ++i) {
        h ^= static_cast<std::uint8_t>(code[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

// The adjugate of a 4 x 4 matrix (row-major) and its determinant, from the
// 2 x 2 minors of its first and last two rows.
double adjugate4(const double* m, double* adj) {
    const double s0 = m[0] * m[5] - m[1] * m[4], s1 = m[0] * m[6] - m[2] * m[4],
                 s2 = m[0] * m[7] - m[3] * m[4], s3 = m[1] * m[6] - m[2] * m[5],
                 s4 = m[1] * m[7] - m[3] * m[5], s5 = m[2] * m[7] - m[3] * m[6];
    const double c5 = m[10] * m[15] - m[11] * m[14], c4 = m[9] * m[15] - m[11] * m[13],
                 c3 = m[9] * m[14] - m[10] * m[13], c2 = m[8] * m[15] - m[11] * m[12],
                 c1 = m[8] * m[14] - m[10] * m[12], c0 = m[8] * m[13] - m[9] * m[12];
    adj[0] = m[5] * c5 - m[6] * c4 + m[7] * c3;
    adj[1] = -m[1] * c5 + m[2] * c4 - m[3] * c3;
    adj[2] = m[13] * s5 - m[14] * s4 + m[15] * s3;
    adj[3] = -m[9] * s5 + m[10] * s4 - m[11] * s3;
    adj[4] = -m[4] * c5 + m[6] * c2 - m[7] * c1;
    adj[5] = m[0] * c5 - m[2] * c2 + m[3] * c1;
    adj[6] = -m[12] * s5 + m[14] * s2 - m[15] * s1;
    adj[7] = m[8] * s5 - m[10] * s2 + m[11] * s1;
    adj[8] = m[4] * c4 - m[5] * c2 + m[7] * c0;
    adj[9] = -m[0] * c4 + m[1] * c2 - m[3] * c0;
    adj[10] = m[12] * s4 - m[13] * s2 + m[15] * s0;
    adj[11] = -m[8] * s4 + m[9] * s2 - m[11] * s0;
    adj[12] = -m[4] * c3 + m[5] * c1 - m[6] * c0;
    adj[13] = m[0] * c3 - m[1] * c1 + m[2] * c0;
    adj[14] = -m[12] * s3 + m[13] * s1 - m[14] * s0;
    adj[15] = m[8] * s3 - m[9] * s1 + m[10] * s0;
    return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
}

// The determinant of a 4 x 4 matrix (row-major), by the same minors.
double determinant4(const double* m) {
    const double s0 = m[0] * m[5] - m[1] * m[4], s1 = m[0] * m[6] - m[2] * m[4],
                 s2 = m[0] * m[7] - m[3] * m[4], s3 = m[1] * m[6] - m[2] * m[5],
                 s4 = m[1] * m[7] - m[3] * m[5], s5 = m[2] * m[7] - m[3] * m[6];
    const double c5 = m[10] * m[15] - m[11] * m[14], c4 = m[9] * m[15] - m[11] * m[13],
                 c3 = m[9] * m[14] - m[10] * m[13], c2 = m[8] * m[15] - m[11] * m[12],
                 c1 = m[8] * m[14] - m[10] * m[12], c0 = m[8] * m[13] - m[9] * m[12];
    return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
}

// The directed edges a walk may start from: those whose nodes and the next
// node of their facet have the largest degrees, a choice that any
// isomorphism keeps.
int starts(int count, const Facet* facets, std::int8_t (*out)[2]) {
    std::int8_t degree[kMaxNeighbors] = {};
    for (int f = 0; f < count; ++f) {
        for (const std::int8_t v : facets[f]) {
            ++degree[v];
        }
    }
    std::uint32_t best = 0;
    int found = 0;
    for (int f = 0; f < count; ++f) {
        for (int e = 0; e < 3; ++e) {
            const int a = facets[f][e], b = facets[f][(e + 1) % 3], c = facets[f][(e + 2) % 3];
            const std::uint32_t key = (static_cast<std::uint32_t>(degree[a]) << 16) |
                                      (static_cast<std::uint32_t>(degree[b]) << 8) |
                                      static_cast<std::uint32_t>(degree[c]);
            if (key > best) {
                best = key;
                found = 0;
            }
            if (key == best) {
                out[found][0] = static_cast<std::int8_t>(a);
                out[found][1] = static_cast<std::int8_t>(b);
                ++found;
            }
        }
    }
    return found;
}

}  // namespace

// ------------------------------------------------------------------ hull

void orient_facet(const Point* points, Facet& facet, const Point& inside, Point& normal) {
    const int a = facet[0], b = facet[1], c = facet[2];
    normal = unit_normal(points, a, b, c);
    Facet f = {facet[0], facet[1], facet[2]};
    if (below(inside, points[a], normal) > 0.0) {  // the normal points outwards
        normal = {-normal[0], -normal[1], -normal[2]};
        f = {facet[1], facet[0], facet[2]};
    }
    const int first = f[0] < f[1] ? (f[0] < f[2] ? 0 : 2) : (f[1] < f[2] ? 1 : 2);
    facet = {f[first], f[(first + 1) % 3], f[(first + 2) % 3]};
}

int grow_hull(Hull& hull, int n, const Point* points, Facet* out) {
    int from = hull.size;
    hull.size = n;
    if (!hull.ok) {
        int corners[4];
        const int error = first_tetrahedron(n, points, corners);
        if (error != 0) {
            return error;
        }
        std::fill(hull.added, hull.added + kMaxPoints, false);
        hull.centre = {0.0, 0.0, 0.0};
        for (const int v : corners) {
            hull.added[v] = true;
            for (int e = 0; e < 3; ++e) {
                hull.centre[e] += points[v][e] / 4.0;
            }
        }
        const int faces[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
        for (int f = 0; f < 4; ++f) {
            hull.facets[f] = {static_cast<std::int8_t>(corners[faces[f][0]]),
                              static_cast<std::int8_t>(corners[faces[f][1]]),
                              static_cast<std::int8_t>(corners[faces[f][2]])};
            orient_facet(points, hull.facets[f], hull.centre, hull.normals[f]);
        }
        hull.count = 4;
        from = 0;
    }
    for (int i = from; i < n; ++i) {
        if (hull.added[i]) {
            continue;
        }
        hull.added[i] = true;
        // the edges between a facet the point sees and one it does not
        // become facets with the point
        std::int8_t seen[kMaxPoints][kMaxPoints] = {};
        Facet fresh[kMaxFacets];
        int fresh_count = 0;
        for (int f = 0; f < hull.count; ++f) {
            const std::int8_t a = hull.facets[f][0], b = hull.facets[f][1], c = hull.facets[f][2];
            const bool visible = below(points[i], points[a], hull.normals[f]) > kFlat;
            const std::int8_t mark = visible ? 1 : 2;
            const int ab = (seen[a][b] |= mark);
            seen[b][a] |= mark;
            const int bc = (seen[b][c] |= mark);
            seen[c][b] |= mark;
            const int ca = (seen[c][a] |= mark);
            seen[a][c] |= mark;
            if (visible) {
                hull.facets[f] = hull.facets[hull.count - 1];
                hull.normals[f] = hull.normals[hull.count - 1];
                --hull.count;
                --f;
            }
            const auto point = static_cast<std::int8_t>(i);
            if (ab == 3 && fresh_count < kMaxFacets) {
                fresh[fresh_count++] = {point, a, b};
            }
            if (bc == 3 && fresh_count < kMaxFacets) {
                fresh[fresh_count++] = {point, b, c};
            }
            if (ca == 3 && fresh_count < kMaxFacets) {
                fresh[fresh_count++] = {point, c, a};
            }
        }
        for (int k = 0; k < fresh_count; ++k) {
            if (hull.count >= kMaxFacets) {
                return -4;
            }
            Facet facet = fresh[k];
            orient_facet(points, facet, hull.centre, hull.normals[hull.count]);
            for (int f = 0; f < hull.count; ++f) {
                if (hull.facets[f] == facet) {
                    return -5;
                }
            }
            hull.facets[hull.count++] = facet;
        }
    }
    for (int f = 0; f < hull.count; ++f) {
        const Facet& facet = hull.facets[f];
        if (facet[0] == 0 || facet[1] == 0 || facet[2] == 0) {
            return 1;
        }
        out[f] = {static_cast<std::int8_t>(facet[0] - 1), static_cast<std::int8_t>(facet[1] - 1),
                  static_cast<std::int8_t>(facet[2] - 1)};
    }
    return 0;
}

int degrees(int count, const Facet* facets, int n, std::int8_t* degree) {
    std::fill(degree, degree + n, std::int8_t{0});
    for (int f = 0; f < count; ++f) {
        for (const std::int8_t v : facets[f]) {
            ++degree[v];
        }
    }
    return n > 0 ? *std::max_element(degree, degree + n) : 0;
}

// ----------------------------------------------------------- graph forms

bool canonical_form(int count, const Facet* facets, int n, const std::int8_t* colours,
                    Form& form) {
    FacetMap map;
    if (!map.build(count, facets)) {
        return false;
    }
    form.length = 3 * count;
    std::fill(form.code.begin(), form.code.end(), static_cast<std::int8_t>(SCHAR_MAX));
    std::int8_t order[kMaxNeighbors];
    std::int8_t edges[3 * kMaxFacets][2];
    const int count_starts = starts(count, facets, edges);
    bool any = false;
    for (int k = 0; k < count_starts; ++k) {
        const int result =
            walk(map, n, form.length, colours, edges[k][0], edges[k][1], form.code.data(), order);
        if (result == -2) {
            return false;
        }
        if (result == 1) {
            std::copy(order, order + n, form.labels.begin());
            any = true;
        }
    }
    form.hash = hash(form.code.data(), form.length);
    return any;
}

std::vector<std::array<std::int8_t, kMaxNeighbors>> canonical_labellings(
    int count, const Facet* facets, int n, const std::int8_t* colours, Form& form) {
    std::vector<std::array<std::int8_t, kMaxNeighbors>> all;
    if (!canonical_form(count, facets, n, colours, form)) {
        return all;
    }
    FacetMap map;
    map.build(count, facets);
    std::int8_t edges[3 * kMaxFacets][2];
    const int count_starts = starts(count, facets, edges);
    for (int k = 0; k < count_starts; ++k) {
        std::array<std::int8_t, 2 * kMaxEdges> code = form.code;
        std::array<std::int8_t, kMaxNeighbors> order{};
        if (walk(map, n, form.length, colours, edges[k][0], edges[k][1], code.data(),
                 order.data()) == 0) {
            all.push_back(order);
        }
    }
    return all;
}

// ------------------------------------------------------------ rotations

void symmetric_eigen(int n, double* a, double* values, double* vectors) {
    for (int i = 0; i < n * n; ++i) {
        vectors[i] = (i / n == i % n) ? 1.0 : 0.0;
    }
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = 0.0, scale = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                (i == j ? scale : off) += a[n * i + j] * a[n * i + j];
            }
        }
        if (off <= 1e-30 * (scale + off) || off == 0.0) {
            break;
        }
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[n * p + q];
                if (apq == 0.0) {
                    continue;
                }
                const double theta = (a[n * q + q] - a[n * p + p]) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {  // columns p and q
                    const double akp = a[n * k + p], akq = a[n * k + q];
                    a[n * k + p] = c * akp - s * akq;
                    a[n * k + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {  // rows p and q
                    const double apk = a[n * p + k], aqk = a[n * q + k];
                    a[n * p + k] = c * apk - s * aqk;
                    a[n * q + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = vectors[n * k + p], vkq = vectors[n * k + q];
                    vectors[n * k + p] = c * vkp - s * vkq;
                    vectors[n * k + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        values[i] = a[n * i + i];
    }
}

Quaternion best_rotation(const double* S, double bound) {
    const double xx = S[0], xy = S[1], xz = S[2], yx = S[3], yy = S[4], yz = S[5], zx = S[6],
                 zy = S[7], zz = S[8];
    // the matrix of Horn: q^T N q = sum_i b_i . R(q) a_i
    double N[16] = {xx + yy + zz, yz - zy,      zx - xz,       xy - yx,
                    yz - zy,      xx - yy - zz, xy + yx,       zx + xz,
                    zx - xz,      xy + yx,      -xx + yy - zz, yz + zy,
                    xy - yx,      zx + xz,      yz + zy,       -xx - yy + zz};
    // its characteristic polynomial l^4 + c2 l^2 + c1 l + c0, by the
    // identities of a traceless N built from S
    double squares = 0.0;
    for (int i = 0; i < 9; ++i) {
        squares += S[i] * S[i];
    }
    const double det_s = S[0] * (S[4] * S[8] - S[5] * S[7]) - S[1] * (S[3] * S[8] - S[5] * S[6]) +
                         S[2] * (S[3] * S[7] - S[4] * S[6]);
    const double c2 = -2.0 * squares, c1 = -8.0 * det_s, c0 = determinant4(N);
    // the largest root, by Newton's method from above
    double lambda = bound;
    if (lambda > 1e-11) {
        for (int it = 0; it < 50; ++it) {
            const double l2 = lambda * lambda;
            const double f = (l2 + c2) * l2 + c1 * lambda + c0;
            const double df = 4.0 * l2 * lambda + 2.0 * c2 * lambda + c1;
            const double delta = f / df;
            lambda -= delta;
            if (std::fabs(delta) < std::fabs(1e-11 * lambda)) {
                break;
            }
        }
    } else {
        lambda = 0.0;
    }
    // its eigenvector: the longest row of the adjugate of N - lambda
    double A[16];
    std::copy(N, N + 16, A);
    for (int i = 0; i < 4; ++i) {
        A[5 * i] -= lambda;
    }
    double adj[16];
    adjugate4(A, adj);
    double longest = -1.0;
    int best = 0;
    for (int i = 0; i < 4; ++i) {
        const double length = adj[4 * i] * adj[4 * i] + adj[4 * i + 1] * adj[4 * i + 1] +
                              adj[4 * i + 2] * adj[4 * i + 2] + adj[4 * i + 3] * adj[4 * i + 3];
        if (length > longest) {
            longest = length;
            best = i;
        }
    }
    const double n2 = 4.0 * squares;  // the sum of the squares of N
    const double scale = n2 > 0.0 ? n2 * n2 * n2 : 1.0;  // ~ the size of an adjugate row squared
    if (!(longest > 1e-20 * scale)) {
        // a repeated largest eigenvalue: by Jacobi rotations
        double values[4], vectors[16];
        symmetric_eigen(4, N, values, vectors);
        const int k = static_cast<int>(std::max_element(values, values + 4) - values);
        return {vectors[k], vectors[4 + k], vectors[8 + k], vectors[12 + k]};
    }
    const double length = std::sqrt(longest);
    return {adj[4 * best] / length, adj[4 * best + 1] / length, adj[4 * best + 2] / length,
            adj[4 * best + 3] / length};
}

void rotation_matrix(const Quaternion& q, double* R) {
    const double a = q[0], b = q[1], c = q[2], d = q[3];
    R[0] = a * a + b * b - c * c - d * d;
    R[1] = 2 * b * c - 2 * a * d;
    R[2] = 2 * b * d + 2 * a * c;
    R[3] = 2 * b * c + 2 * a * d;
    R[4] = a * a - b * b + c * c - d * d;
    R[5] = 2 * c * d - 2 * a * b;
    R[6] = 2 * b * d - 2 * a * c;
    R[7] = 2 * c * d + 2 * a * b;
    R[8] = a * a - b * b - c * c + d * d;
}

Quaternion quaternion(const double* R) {
    // the largest of 4 w^2, 4 x^2, 4 y^2, 4 z^2 decides the formula
    const double trace = R[0] + R[4] + R[8];
    const double squares[4] = {1 + trace, 1 + R[0] - R[4] - R[8], 1 - R[0] + R[4] - R[8],
                               1 - R[0] - R[4] + R[8]};
    const int k = static_cast<int>(std::max_element(squares, squares + 4) - squares);
    const double s = 2.0 * std::sqrt(squares[k]);
    Quaternion q;
    switch (k) {
        case 0:
            q = {s / 4, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s};
            break;
        case 1:
            q = {(R[7] - R[5]) / s, s / 4, (R[1] + R[3]) / s, (R[2] + R[6]) / s};
            break;
        case 2:
            q = {(R[2] - R[6]) / s, (R[1] + R[3]) / s, s / 4, (R[5] + R[7]) / s};
            break;
        default:
            q = {(R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, s / 4};
    }
    if (q[0] < 0.0) {
        q = {-q[0], -q[1], -q[2], -q[3]};
    }
    return q;
}

Quaternion product(const Quaternion& a, const Quaternion& b) {
    return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}

// ------------------------------------------------------ correspondences

void complete_order(int n, int size, std::int8_t* order) {
    bool hit[kMaxInput] = {};
    for (int i = 0; i < n; ++i) {
        hit[order[i]] = true;
    }
    int next = n;
    for (int i = 0; i < size; ++i) {
        if (!hit[i]) {
            order[next++] = static_cast<std::int8_t>(i);
        }
    }
}

std::uint64_t encode_permutation(int base, int k, const std::int8_t* permutation) {
    int place[kMaxInput], at[kMaxInput];  // place of each number; number at each place
    for (int i = 0; i < base; ++i) {
        place[i] = at[i] = i;
    }
    int digits[kMaxInput] = {};
    for (int i = 0; i < k; ++i) {
        const int e = permutation[i];
        const int j = place[e];
        digits[i] = j - i;
        if (j > i) {
            std::swap(place[at[i]], place[at[j]]);
            std::swap(at[i], at[j]);
        }
    }
    std::uint64_t code = 0;
    for (int t = 0; t < k; ++t) {
        code = code * static_cast<std::uint64_t>(1 + base - k + t) +
               static_cast<std::uint64_t>(digits[k - 1 - t]);
    }
    return code;
}

void decode_permutation(int base, int k, std::uint64_t code, std::int8_t* permutation) {
    int digits[kMaxInput] = {};
    for (int i = 0; i < base; ++i) {
        digits[i] = static_cast<int>(code % static_cast<std::uint64_t>(base - i));
        code /= static_cast<std::uint64_t>(base - i);
    }
    std::int8_t numbers[kMaxInput];
    for (int i = 0; i < base; ++i) {
        numbers[i] = static_cast<std::int8_t>(i);
    }
    for (int i = 0; i < std::min(k, base); ++i) {
        std::swap(numbers[i], numbers[i + digits[i]]);
        permutation[i] = numbers[i];
    }
}

}  // namespace mdapy::identify::ptm

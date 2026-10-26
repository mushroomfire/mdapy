// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "identify/ptm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/buffer.h"
#include "core/voronoi_cell.h"
#include "identify/ptm_data.h"
#include "identify/ptm_tools.h"
#include "neighbor/frame.h"

namespace mdapy::identify {
namespace {

using ptm::Facet;
using ptm::Form;
using ptm::Hull;
using ptm::kMaxFacets;
using ptm::kMaxInput;
using ptm::kMaxNeighbors;
using ptm::kMaxPoints;
using ptm::Point;
using ptm::Quaternion;
using ptm::Reference;

// The neighbors that the shells of diamond and graphene are drawn from: the
// nearest of each atom.
constexpr int kShellReach = 13;

// An atom and its neighbors: their atoms, chemical types and vectors from
// it, and for each its place among the nearest neighbors of the atom it was
// found from, from 1 (0 for the atom itself).
struct Environment {
    int count = 0;
    std::int8_t ranks[kMaxInput];
    std::int64_t atoms[kMaxInput];
    std::int32_t types[kMaxInput];
    Point points[kMaxInput];
};

// The best fit so far: of which structure, its rmsd, scale and rotation, and
// the neighbor that fills each point of the template.
struct Fit {
    const Reference* ref = nullptr;
    double rmsd = std::numeric_limits<double>::infinity();
    double scale = 0.0;
    Quaternion q = {1.0, 0.0, 0.0, 0.0};
    std::int8_t mapping[kMaxPoints];
};

double dot(const Point& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// ------------------------------------------------------------- neighbors

// The order of the first count neighbors (vectors) of an atom: the larger
// the solid angle of their face of its Voronoi cell, the earlier, then the
// nearer; coded as a permutation of 18.
std::uint64_t voronoi_order(geometry::Cell& cell, int count, const Point* vectors) {
    double norms2[kMaxInput], largest = 0.0;
    for (int k = 0; k < count; ++k) {
        norms2[k] = dot(vectors[k], vectors[k]);
        largest = std::max(largest, norms2[k]);
    }
    largest = std::sqrt(largest);
    cell.set_tolerance(1e-11 * largest);
    cell.reset(10.0 * largest);
    for (int k = 0; k < count; ++k) {
        const double length = std::sqrt(norms2[k]);
        const geometry::Vec normal = {vectors[k][0] / length, vectors[k][1] / length,
                                      vectors[k][2] / length};
        cell.cut(normal, 0.5 * length, k);
    }
    // the solid angle of each face seen from the atom (Van Oosterom and
    // Strackee 1983), a fan of triangles from its first corner
    double angles[kMaxInput] = {};
    const auto& points = cell.points();
    auto unit = [&](std::uint32_t v) {
        const geometry::Vec& p = points[v];
        const double length = std::sqrt(geometry::dot(p, p));
        return geometry::Vec{p[0] / length, p[1] / length, p[2] / length};
    };
    for (const geometry::Face& face : cell.faces()) {
        if (face.id < 0 || face.id >= count) {
            continue;
        }
        const std::uint32_t* corners = cell.corners(face);
        const geometry::Vec a = unit(corners[0]);
        geometry::Vec b = unit(corners[1]);
        double sum = 0.0;
        for (std::size_t k = 2; k < face.count; ++k) {
            const geometry::Vec c = unit(corners[k]);
            const double top = geometry::dot(a, geometry::cross(b, c));
            const double bottom =
                1.0 + geometry::dot(a, b) + geometry::dot(b, c) + geometry::dot(c, a);
            sum += std::fabs(2.0 * std::atan2(top, bottom));
            b = c;
        }
        angles[face.id] = sum;
    }
    std::int8_t order[kMaxInput];
    for (int k = 0; k < count; ++k) {
        order[k] = static_cast<std::int8_t>(k);
    }
    std::stable_sort(order, order + count, [&](std::int8_t x, std::int8_t y) {
        if (angles[x] != angles[y]) {
            return angles[x] > angles[y];
        }
        return norms2[x] < norms2[y];
    });
    ptm::complete_order(count, kMaxInput - 1, order);
    return ptm::encode_permutation(kMaxInput - 1, kMaxInput - 1, order);
}

// An atom and its neighbors in the Voronoi order of the first pass.
void environment(const PtmInput& in, const std::uint64_t* orders, std::int64_t i,
                 Environment& env) {
    const std::int64_t first = in.offsets[i];
    const int count =
        static_cast<int>(std::min<std::int64_t>(in.offsets[i + 1] - first, kMaxInput - 1));
    std::int8_t order[kMaxInput];
    ptm::decode_permutation(kMaxInput - 1, kMaxInput - 1, orders[i], order);
    env.ranks[0] = 0;
    env.atoms[0] = i;
    env.types[0] = in.types ? in.types[i] : 0;
    env.points[0] = {0.0, 0.0, 0.0};
    const double* xi = in.positions + 3 * i;
    for (int k = 0; k < count; ++k) {
        const std::int64_t p = first + order[k];
        const auto j = static_cast<std::int64_t>(in.j[p]);
        neighbor::pair_vector(xi, in.positions + 3 * j, in.shifts + 3 * p, in.matrix,
                              env.points[k + 1].data());
        env.ranks[k + 1] = static_cast<std::int8_t>(order[k] + 1);
        env.atoms[k + 1] = j;
        env.types[k + 1] = in.types ? in.types[j] : 0;
    }
    env.count = count + 1;
}

// Keeps the neighbors among the nearest kShellReach of their atom.
void keep_nearest(Environment& env) {
    int kept = 0;
    for (int k = 0; k < env.count; ++k) {
        if (env.ranks[k] <= kShellReach) {
            env.ranks[kept] = env.ranks[k];
            env.atoms[kept] = env.atoms[k];
            env.types[kept] = env.types[k];
            env.points[kept] = env.points[k];
            ++kept;
        }
    }
    env.count = kept;
}

// The two shells of diamond (4 nearest neighbors, 3 more of each) or
// graphene (3 and 2): the inner shell the first of the atom's neighbors,
// the outer shell the first neighbors of each inner one that are neither
// the atom, an inner neighbor nor already taken, by turns in their order.
bool two_shells(const PtmInput& in, const std::uint64_t* orders, std::int64_t i, int inner,
                int outer, const Environment* known, Environment& out) {
    Environment env;
    if (known != nullptr && known->count >= inner) {
        env = *known;
    } else {
        environment(in, orders, i, env);
    }
    keep_nearest(env);
    if (env.count < inner + 1) {
        return false;
    }
    for (int k = 0; k <= inner; ++k) {
        out.ranks[k] = env.ranks[k];
        out.atoms[k] = env.atoms[k];
        out.types[k] = env.types[k];
        out.points[k] = env.points[k];
    }
    const Point& d1 = out.points[1];
    const double tolerance = std::max(1e-5 * std::sqrt(dot(d1, d1)), 1e-5);
    struct Candidate {
        int place, inner;
        std::int8_t rank;
        std::int64_t atom;
        std::int32_t type;
        Point delta;
    };
    std::vector<Candidate> candidates;
    for (int a = 0; a < inner; ++a) {
        Environment second;
        environment(in, orders, out.atoms[1 + a], second);
        keep_nearest(second);
        if (second.count < inner + 1) {
            return false;
        }
        const Point& q = out.points[1 + a];
        for (int k = 1; k < second.count; ++k) {
            const Point& p = second.points[k];
            candidates.push_back({k, a, second.ranks[k], second.atoms[k], second.types[k],
                                  {p[0] + q[0], p[1] + q[1], p[2] + q[2]}});
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& x, const Candidate& y) { return x.place < y.place; });
    int counts[4] = {0, 0, 0, 0};
    auto taken = [&](const Candidate& c) {
        auto same = [&](int slot) {
            if (out.atoms[slot] != c.atom) {
                return false;
            }
            const Point& p = out.points[slot];
            const Point d = {p[0] - c.delta[0], p[1] - c.delta[1], p[2] - c.delta[2]};
            return std::sqrt(dot(d, d)) < tolerance;
        };
        for (int slot = 0; slot <= inner; ++slot) {
            if (same(slot)) {
                return true;
            }
        }
        for (int a = 0; a < inner; ++a) {
            for (int k = 0; k < counts[a]; ++k) {
                if (same(1 + inner + outer * a + k)) {
                    return true;
                }
            }
        }
        return false;
    };
    int found = 0;
    for (const Candidate& c : candidates) {
        if (counts[c.inner] >= outer || taken(c)) {
            continue;
        }
        const int slot = 1 + inner + outer * c.inner + counts[c.inner];
        out.ranks[slot] = c.rank;
        out.atoms[slot] = c.atom;
        out.types[slot] = c.type;
        out.points[slot] = c.delta;
        ++counts[c.inner];
        if (++found >= inner * outer) {
            break;
        }
    }
    out.count = 1 + inner + inner * outer;
    return found == inner * outer;
}

// ------------------------------------------------------------- matching

// The points less their centre.
void centred(int n, const Point* points, Point* out) {
    Point centre = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        for (int e = 0; e < 3; ++e) {
            centre[e] += points[i][e];
        }
    }
    for (int e = 0; e < 3; ++e) {
        centre[e] /= n;
    }
    for (int i = 0; i < n; ++i) {
        out[i] = {points[i][0] - centre[0], points[i][1] - centre[1], points[i][2] - centre[2]};
    }
}

// The same, scaled so that the distances of all but the first from the
// centre add up to n.
void normalized(int n, const Point* points, Point* out) {
    centred(n, points, out);
    double mean = 0.0;
    for (int i = 1; i < n; ++i) {
        mean += std::sqrt(dot(out[i], out[i]));
    }
    mean /= n;
    for (int i = 0; i < n; ++i) {
        for (int e = 0; e < 3; ++e) {
            out[i][e] /= mean;
        }
    }
}

// Keeps a correspondence if its fit is the best so far: the template turned
// onto the points by the best rotation, then scaled.
void consider(const Reference& ref, const std::int8_t* mapping, const Point* observed, double g1,
              double g2, Fit& fit) {
    const int n = ref.neighbors + 1;
    const auto& ideal = ref.templates[0];
    double S[9] = {};
    for (int i = 0; i < n; ++i) {
        const Point& a = ideal[i];
        const Point& b = observed[mapping[i]];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                S[3 * r + c] += a[r] * b[c];
            }
        }
    }
    const Quaternion q = ptm::best_rotation(S, 0.5 * (g1 + g2));
    double R[9];
    ptm::rotation_matrix(q, R);
    double k0 = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int r = 0; r < 3; ++r) {
            const double v =
                R[3 * r] * ideal[i][0] + R[3 * r + 1] * ideal[i][1] + R[3 * r + 2] * ideal[i][2];
            k0 += v * observed[mapping[i]][r];
        }
    }
    const double scale = k0 / g2;
    const double rmsd = std::sqrt(std::fabs(g1 - scale * k0) / n);
    if (rmsd < fit.rmsd) {
        fit.rmsd = rmsd;
        fit.scale = scale;
        fit.ref = &ref;
        fit.q = q;
        std::copy(mapping, mapping + n, fit.mapping);
    }
}

void sums(const Reference& ref, const Point* observed, double& g1, double& g2) {
    g1 = g2 = 0.0;
    for (int i = 0; i <= ref.neighbors; ++i) {
        g1 += dot(ref.templates[0][i], ref.templates[0][i]);
        g2 += dot(observed[i], observed[i]);
    }
}

// Tries the correspondences that the graphs of ref with the canonical form
// of the hull of the neighbors give.
void try_graphs(const Reference& ref, const Form& form, const Point* observed, Fit& fit) {
    double g1, g2;
    sums(ref, observed, g1, g2);
    std::int8_t node[kMaxNeighbors];  // place in the canonical walk -> neighbor
    for (int v = 0; v < ref.neighbors; ++v) {
        node[form.labels[v]] = static_cast<std::int8_t>(v);
    }
    std::int8_t mapping[kMaxPoints];
    mapping[0] = 0;
    for (const ptm::Graph& graph : ref.graphs) {
        if (graph.form.hash != form.hash || graph.form.length != form.length ||
            !std::equal(form.code.begin(), form.code.begin() + form.length,
                        graph.form.code.begin())) {
            continue;
        }
        for (const auto& labels : graph.labellings) {
            for (int t = 0; t < ref.neighbors; ++t) {
                mapping[t + 1] = static_cast<std::int8_t>(node[labels[t]] + 1);
            }
            consider(ref, mapping, observed, g1, g2, fit);
        }
    }
}

// sc or bcc: the hull of the first neighbors must be that of the template.
void match_hull(const Reference& ref, const Point* hull_points, const Point* points, Hull& hull,
                Fit& fit) {
    const int n = ref.neighbors;
    Facet facets[kMaxFacets];
    const bool was_ok = hull.ok;
    int result = ptm::grow_hull(hull, n + 1, hull_points, facets);
    hull.ok = result >= 0;
    if (was_ok && !hull.ok) {  // a hull grown from fewer points can fail where a new one does not
        result = ptm::grow_hull(hull, n + 1, hull_points, facets);
        hull.ok = result >= 0;
    }
    if (result != 0 || hull.count != ref.facets) {
        return;
    }
    std::int8_t degree[kMaxNeighbors];
    if (ptm::degrees(hull.count, facets, n, degree) > ref.max_degree) {
        return;
    }
    if (ref.type == ptm::kSc &&
        std::any_of(degree, degree + n, [](std::int8_t d) { return d != 4; })) {
        return;
    }
    Point observed[kMaxPoints];
    centred(n + 1, points, observed);
    const std::int8_t colours[kMaxNeighbors] = {};
    Form form;
    if (ptm::canonical_form(hull.count, facets, n, colours, form)) {
        try_graphs(ref, form, observed, fit);
    }
}

// fcc, hcp and ico share the hull of the 12 nearest neighbors.
void match_close_packed(std::int32_t structures, const Point* hull_points, const Point* points,
                        Hull& hull, Fit& fit) {
    const Reference& fcc = ptm::reference(ptm::kFcc);
    const int n = fcc.neighbors;
    Facet facets[kMaxFacets];
    const int result = ptm::grow_hull(hull, n + 1, hull_points, facets);
    hull.ok = result >= 0;
    if (result != 0 || hull.count != fcc.facets) {
        return;
    }
    std::int8_t degree[kMaxNeighbors];
    if (ptm::degrees(hull.count, facets, n, degree) > fcc.max_degree) {
        return;
    }
    Point observed[kMaxPoints];
    centred(n + 1, points, observed);
    const std::int8_t colours[kMaxNeighbors] = {};
    Form form;
    if (!ptm::canonical_form(hull.count, facets, n, colours, form)) {
        return;
    }
    for (const int type : {ptm::kFcc, ptm::kHcp, ptm::kIco}) {
        if (structures & (1 << (type - 1))) {
            try_graphs(ptm::reference(type), form, observed, fit);
        }
    }
}

// Cubic and hexagonal diamond: the hull of the 16 neighbors, the facet of
// the three outer neighbors of an inner one turned into three facets with
// it, the inner neighbors in their own colour.
void match_diamond(std::int32_t structures, const Point* hull_points, const Point* points,
                   Fit& fit) {
    const Reference& dcub = ptm::reference(ptm::kCubicDiamond);
    const int n = dcub.neighbors;
    const int outer_facets = ptm::reference(ptm::kFcc).facets;  // a hull of 12 points
    Hull hull;
    Facet facets[kMaxFacets];
    if (ptm::grow_hull(hull, n + 1, hull_points, facets) != 0) {
        return;
    }
    int count = hull.count;
    // an inner neighbor on the hull: on no facet with another
    bool on_hull[4] = {false, false, false, false};
    for (int f = 0; f < count; ++f) {
        int inner = 0;
        for (const std::int8_t v : facets[f]) {
            if (v <= 3) {
                on_hull[v] = true;
                ++inner;
            }
        }
        if (inner > 1) {
            return;
        }
    }
    const int out = static_cast<int>(std::count(on_hull, on_hull + 4, true));
    if (count != outer_facets + 2 * out) {
        return;
    }
    std::int8_t degree[kMaxNeighbors];
    if (ptm::degrees(outer_facets, facets, n, degree) > dcub.max_degree) {
        return;
    }
    int tents = 0;
    Facet tented[4];
    for (int f = 0; f < count; ++f) {
        const int a = facets[f][0], b = facets[f][1], c = facets[f][2];
        if (a <= 3 || b <= 3 || c <= 3) {
            continue;
        }
        const int group = (a - 4) / 3;
        if (group == (b - 4) / 3 && group == (c - 4) / 3) {
            if (tents + out >= 4) {
                return;
            }
            tented[tents++] = facets[f];
            facets[f] = facets[count - 1];
            --count;
            --f;
        }
    }
    if (tents + out != 4) {
        return;
    }
    for (int t = 0; t < tents; ++t) {
        const std::int8_t a = tented[t][0], b = tented[t][1], c = tented[t][2];
        const auto inner = static_cast<std::int8_t>((a - 4) / 3);
        facets[count++] = {inner, b, c};
        facets[count++] = {a, inner, c};
        facets[count++] = {a, b, inner};
    }
    if (ptm::degrees(count, facets, n, degree) > dcub.max_degree) {
        return;
    }
    Point observed[kMaxPoints];
    centred(n + 1, points, observed);
    const std::int8_t colours[kMaxNeighbors] = {1, 1, 1, 1};
    Form form;
    if (!ptm::canonical_form(count, facets, n, colours, form)) {
        return;
    }
    for (const int type : {ptm::kCubicDiamond, ptm::kHexagonalDiamond}) {
        if (structures & (1 << (type - 1))) {
            try_graphs(ptm::reference(type), form, observed, fit);
        }
    }
}

// Graphene: the outer neighbors of each inner one in either order.
void match_graphene(const Point* points, Fit& fit) {
    const Reference& ref = ptm::reference(ptm::kGraphene);
    const int n = ref.neighbors + 1;
    Point observed[kMaxPoints];
    centred(n, points, observed);
    double g1, g2;
    sums(ref, observed, g1, g2);
    std::int8_t mapping[kMaxPoints];
    for (int i = 0; i < n; ++i) {
        mapping[i] = static_cast<std::int8_t>(i);
    }
    for (int x = 0; x < 2; ++x) {
        std::swap(mapping[4], mapping[5]);
        for (int y = 0; y < 2; ++y) {
            std::swap(mapping[6], mapping[7]);
            for (int z = 0; z < 2; ++z) {
                std::swap(mapping[8], mapping[9]);
                consider(ref, mapping, observed, g1, g2, fit);
            }
        }
    }
}

// ---------------------------------------------------------------- output

// The chemical ordering of the neighbors of an atom of the matched
// structure.
int ordering(const Reference& ref, const std::int8_t* mapping, const std::int32_t* types) {
    const int n = ref.neighbors + 1;
    const std::int32_t centre = types[0];
    bool pure = true, binary = true, second_seen = false;
    std::int32_t second = 0;
    for (int i = 1; i < n; ++i) {
        const std::int32_t t = types[i];
        if (t == centre) {
            continue;
        }
        pure = false;
        if (!second_seen) {
            second = t;
            second_seen = true;
        } else if (t != second) {
            binary = false;
        }
    }
    if (pure) {
        return ptm::kPure;
    }
    if (!binary) {
        return ptm::kOrderingNone;
    }
    std::int8_t other[kMaxPoints];
    for (int i = 0; i < n; ++i) {
        other[i] = static_cast<std::int8_t>(types[mapping[i]] != centre);
    }
    if (ref.type == ptm::kFcc) {
        const std::uint32_t code = ptm::canonical_pattern(ref, other);
        const ptm::FccOrderings& fcc = ptm::fcc_orderings();
        if (code == fcc.l10) return ptm::kL10;
        if (code == fcc.l12_majority) return ptm::kL12Majority;
        if (code == fcc.l12_minority) return ptm::kL12Minority;
    }
    // an inner shell all of the other element, the rest of the first
    auto shells = [&](int inner) {
        for (int i = 1; i < n; ++i) {
            if (other[i] != (i <= inner ? 1 : 0)) {
                return false;
            }
        }
        return true;
    };
    if (ref.type == ptm::kBcc && shells(8)) return ptm::kB2;
    if ((ref.type == ptm::kCubicDiamond || ref.type == ptm::kHexagonalDiamond) && shells(4)) {
        return ptm::kZincblende;
    }
    if (ref.type == ptm::kGraphene && shells(3)) return ptm::kBoronNitride;
    return ptm::kOrderingNone;
}

// The code of the neighbors in the order of the template, as OVITO stores
// it: their places among the nearest of their atoms, the template in the
// top two bits.
std::uint64_t correspondences(const Reference& ref, const std::int8_t* ranks, int template_index) {
    const int n = ref.neighbors;
    std::uint64_t code;
    if (ref.type == ptm::kCubicDiamond || ref.type == ptm::kHexagonalDiamond ||
        ref.type == ptm::kGraphene) {
        const int inner = ref.type == ptm::kGraphene ? 3 : 4;
        const int outer = ref.type == ptm::kGraphene ? 2 : 3;
        std::int8_t places[kMaxNeighbors];
        for (int i = 0; i < n; ++i) {
            places[i] = static_cast<std::int8_t>(ranks[i + 1] - 1);
        }
        code = ptm::encode_permutation(kShellReach, inner, places);
        for (int a = 0; a < inner; ++a) {
            code |= ptm::encode_permutation(kShellReach, outer, places + inner + outer * a)
                    << (15 + 11 * a);
        }
    } else {
        std::int8_t places[kMaxInput];
        for (int i = 0; i <= n; ++i) {
            places[i] = ranks[i];
        }
        ptm::complete_order(n + 1, kMaxInput, places);
        for (int i = 0; i < kMaxInput - 1; ++i) {
            places[i] = static_cast<std::int8_t>(places[i + 1] - 1);
        }
        code = ptm::encode_permutation(kMaxInput - 1, kMaxInput - 1, places);
    }
    return code | (static_cast<std::uint64_t>(template_index) << 62);
}

// Writes the results of atom i.
void write(const PtmInput& in, const PtmOutput& out, std::int64_t i, const Fit& fit,
           const Environment& env) {
    const bool found = fit.ref != nullptr;
    if (out.structure) out.structure[i] = static_cast<std::int8_t>(found ? fit.ref->type : 0);
    if (out.rmsd) out.rmsd[i] = found ? fit.rmsd : 0.0;
    if (out.interatomic_distance) {
        out.interatomic_distance[i] = found ? fit.ref->distance / fit.scale : 0.0;
    }
    if (!found) {
        if (out.orientation) std::fill(out.orientation + 4 * i, out.orientation + 4 * i + 4, 0.0);
        if (out.deformation) std::fill(out.deformation + 9 * i, out.deformation + 9 * i + 9, 0.0);
        if (out.ordering) out.ordering[i] = 0;
        if (out.correspondences) out.correspondences[i] = 0;
        if (out.template_index) out.template_index[i] = 0;
        if (out.neighbors) {
            std::fill(out.neighbors + kMaxNeighbors * i, out.neighbors + kMaxNeighbors * (i + 1),
                      static_cast<index_t>(-1));
        }
        return;
    }
    const Reference& ref = *fit.ref;
    const int n = ref.neighbors + 1;
    if (out.ordering) {
        out.ordering[i] = static_cast<std::int8_t>(ordering(ref, fit.mapping, env.types));
    }
    // the orientation into the fundamental zone, with the template and the
    // order of the neighbors that go with it
    Quaternion q = fit.q;
    const auto g = static_cast<std::size_t>(ptm::into_fundamental_zone(*ref.group, q));
    const int template_index = ref.group_template[g];
    std::int8_t mapping[kMaxPoints];
    for (int j = 0; j < n; ++j) {
        mapping[j] = fit.mapping[ref.moves[g][j]];
    }
    if (out.orientation) {
        double* o = out.orientation + 4 * i;
        o[0] = q[1];
        o[1] = q[2];
        o[2] = q[3];
        o[3] = q[0];
    }
    if (out.deformation) {
        double* F = out.deformation + 9 * i;
        if (in.deformation) {
            Point scaled[kMaxInput];
            centred(n, env.points, scaled);
            const auto& penrose = ref.penrose[static_cast<std::size_t>(template_index)];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    double sum = 0.0;
                    for (int k = 0; k < n; ++k) {
                        sum += penrose[k][c] * (scaled[mapping[k]][r] * fit.scale);
                    }
                    F[3 * r + c] = sum;
                }
            }
            if (ref.type == ptm::kGraphene) {
                F[8] = 1.0;  // flat: no stretch across the sheet
            }
        } else {
            std::fill(F, F + 9, 0.0);
        }
    }
    if (out.template_index) out.template_index[i] = static_cast<std::int8_t>(template_index);
    if (out.correspondences) {
        std::int8_t ranks[kMaxInput];
        for (int j = 0; j < n; ++j) {
            ranks[j] = env.ranks[mapping[j]];
        }
        out.correspondences[i] = correspondences(ref, ranks, template_index);
    }
    if (out.neighbors) {
        for (int k = 0; k < kMaxNeighbors; ++k) {
            out.neighbors[kMaxNeighbors * i + k] =
                k < ref.neighbors ? static_cast<index_t>(env.atoms[mapping[k + 1]])
                                  : static_cast<index_t>(-1);
        }
    }
}

}  // namespace

void polyhedral_template_matching(const PtmInput& in, const PtmOutput& out,
                                  const Control& control) {
    const std::int64_t n = in.n;
    constexpr std::int64_t kGrain = 512;
    ptm::reference(ptm::kFcc);  // built once, before the threads need it
    ptm::fcc_orderings();
    // the neighbors of each atom ordered by the faces of its Voronoi cell
    Buffer<std::uint64_t> orders(static_cast<std::size_t>(n));  // every atom writes its own
    parallel_for(
        n, kGrain,
        [&](std::int64_t begin, std::int64_t end) {
            geometry::Cell cell(0.0);
            Point vectors[kMaxInput];
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t first = in.offsets[i];
                const int count = static_cast<int>(
                    std::min<std::int64_t>(in.offsets[i + 1] - first, kMaxInput - 1));
                for (int k = 0; k < count; ++k) {
                    const auto j = static_cast<std::int64_t>(in.j[first + k]);
                    neighbor::pair_vector(in.positions + 3 * i, in.positions + 3 * j,
                                          in.shifts + 3 * (first + k), in.matrix,
                                          vectors[k].data());
                }
                orders[static_cast<std::size_t>(i)] = voronoi_order(cell, count, vectors);
            }
        },
        control);
    const std::int32_t s = in.structures;
    auto wanted = [&](int type) { return (s & (1 << (type - 1))) != 0; };
    const bool close = wanted(ptm::kFcc) || wanted(ptm::kHcp) || wanted(ptm::kIco);
    const bool single = close || wanted(ptm::kSc) || wanted(ptm::kBcc);
    const bool diamond = wanted(ptm::kCubicDiamond) || wanted(ptm::kHexagonalDiamond);
    const bool graphene = wanted(ptm::kGraphene);
    parallel_for(
        n, kGrain,
        [&](std::int64_t begin, std::int64_t end) {
            Environment env, diamond_env, graphene_env;
            Point hull_points[kMaxInput];
            for (std::int64_t i = begin; i < end; ++i) {
                Fit fit;
                env.count = 0;
                if (single) {
                    environment(in, orders.data(), i, env);
                    const int count = env.count;
                    normalized(count, env.points, hull_points);
                    Hull hull;
                    if (wanted(ptm::kSc) && count >= 7) {
                        match_hull(ptm::reference(ptm::kSc), hull_points, env.points, hull, fit);
                    }
                    if (close && count >= 13) {
                        match_close_packed(s, hull_points, env.points, hull, fit);
                    }
                    if (wanted(ptm::kBcc) && count >= 15) {
                        match_hull(ptm::reference(ptm::kBcc), hull_points, env.points, hull, fit);
                    }
                }
                if (diamond && two_shells(in, orders.data(), i, 4, 3, &env, diamond_env)) {
                    normalized(17, diamond_env.points, hull_points);
                    match_diamond(s, hull_points, diamond_env.points, fit);
                }
                if (graphene && two_shells(in, orders.data(), i, 3, 2, &env, graphene_env)) {
                    match_graphene(graphene_env.points, fit);
                }
                const Environment* chosen = &env;
                if (fit.ref != nullptr) {
                    const int type = fit.ref->type;
                    if (type == ptm::kCubicDiamond || type == ptm::kHexagonalDiamond) {
                        chosen = &diamond_env;
                    } else if (type == ptm::kGraphene) {
                        chosen = &graphene_env;
                    }
                }
                write(in, out, i, fit, *chosen);
            }
        },
        control);
}

}  // namespace mdapy::identify

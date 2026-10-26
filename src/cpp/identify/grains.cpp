// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "identify/grains.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

#include "identify/ptm.h"
#include "identify/ptm_data.h"
#include "identify/ptm_tools.h"

namespace mdapy::identify {
namespace {

using ptm::Quaternion;  // w, x, y, z

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kMax = std::numeric_limits<double>::max();
constexpr double kDegrees = 180.0 / 3.14159265358979323846;
// bonds misoriented by more are not part of a grain
constexpr double kMisorientationCutoff = 4.0;  // degrees
// merges of smaller clusters are left out of the plots
constexpr std::int64_t kMinPlotSize = 20;
// neighbors of an atom of no structure taken into account
constexpr int kDisorderedNeighbors = 8;

struct Bond {
    std::int64_t a, b;
    double disorientation;
    double length;
};

struct Merge {
    std::int64_t a, b;
    double distance;
    double disorientation;
    std::int64_t size;
    double merge_size = 0.0;
    Quaternion orientation;
};

// Calls the monitor and stops when cancelled.
void checkpoint(const Control& control, std::int64_t done, std::int64_t total) {
    if (control.cancel != nullptr && control.cancel->cancelled()) {
        throw Cancelled();
    }
    if (control.monitor && !control.monitor(done, total)) {
        throw Cancelled();
    }
}

bool cubic(int type) {
    return type == kPtmSc || type == kPtmFcc || type == kPtmBcc || type == kPtmCubicDiamond;
}

bool hexagonal(int type) {
    return type == kPtmHcp || type == kPtmHexagonalDiamond || type == kPtmGraphene;
}

const std::vector<Quaternion>& cubic_group() { return *ptm::reference(ptm::kFcc).group; }
const std::vector<Quaternion>& hexagonal_group() { return *ptm::reference(ptm::kHcp).group; }

const std::vector<Quaternion>* group_of(int type) {
    if (cubic(type)) {
        return &cubic_group();
    }
    if (hexagonal(type)) {
        return &hexagonal_group();
    }
    if (type == kPtmIco) {
        return ptm::reference(ptm::kIco).group;
    }
    return nullptr;
}

Quaternion conjugate(const Quaternion& q) { return {q[0], -q[1], -q[2], -q[3]}; }

// The angle between two orientations in radians, the smallest over the
// symmetry of the lattice.
double disorientation(const Quaternion& q0, const Quaternion& q1, const std::vector<Quaternion>& group) {
    Quaternion turn = ptm::product(conjugate(q0), q1);
    ptm::into_fundamental_zone(group, turn);
    const double t = std::clamp(turn[0], -1.0, 1.0);
    return std::acos(2 * t * t - 1);
}

// The disorientation in degrees of two atoms of the same structure.
double bond_disorientation(int type_a, int type_b, const Quaternion& qa, const Quaternion& qb) {
    if (type_a != type_b) {
        return kMax;
    }
    if (cubic(type_a)) {
        return disorientation(qa, qb, cubic_group()) * kDegrees;
    }
    if (hexagonal(type_a)) {
        return disorientation(qa, qb, hexagonal_group()) * kDegrees;
    }
    return kMax;
}

// The two rotations taking the conventional hcp frame (c along z) to the
// frames of an fcc lattice whose (111) planes are its basal planes; with
// the first component negated, the rotations of the other direction.
const std::array<Quaternion, 2>& hexagonal_to_cubic_turns() {
    static const std::array<Quaternion, 2> turns = [] {
        const double r2 = std::sqrt(2.0), r3 = std::sqrt(3.0), r6 = std::sqrt(6.0);
        return std::array<Quaternion, 2>{
            Quaternion{std::sqrt(-r6 / 3 - r2 / 2 + r3 / 3 + 1) / 2, std::sqrt(-r2 / 2 - r3 / 3 + r6 / 3 + 1) / 2,
                       std::sqrt(-r6 / 3 - r3 / 3 + r2 / 2 + 1) / 2, std::sqrt(r3 / 3 + r2 / 2 + r6 / 3 + 1) / 2},
            Quaternion{std::sqrt(-r3 / 3 + r6 / 6 + 1) / 2, -std::sqrt(-r6 / 6 + r3 / 3 + 1) / 2,
                       std::sqrt(r6 / 6 + r3 / 3 + 1) / 2, -std::sqrt(-r3 / 3 - r6 / 6 + 1) / 2}};
    }();
    return turns;
}

// A hexagonal orientation turned into the cubic orientation nearest a
// cubic one: returns the disorientation in radians, the orientation in q.
double hexagonal_to_cubic(const Quaternion& cubic_q, Quaternion& q) {
    double best = kInfinity;
    Quaternion closest = q;
    for (const Quaternion& turn : hexagonal_to_cubic_turns()) {
        const Quaternion turned = ptm::product(q, turn);
        const double d = disorientation(cubic_q, turned, cubic_group());
        if (d < best) {
            closest = turned;
            best = d;
        }
    }
    q = closest;
    ptm::into_fundamental_zone(cubic_group(), q);
    return best;
}

// A cubic orientation turned into the hexagonal orientation nearest a
// hexagonal one.
double cubic_to_hexagonal(const Quaternion& hexagonal_q, Quaternion& q) {
    double best = kInfinity;
    Quaternion closest = q;
    for (const Quaternion& g : cubic_group()) {
        const Quaternion adjusted = ptm::product(q, g);
        for (Quaternion turn : hexagonal_to_cubic_turns()) {
            turn[0] = -turn[0];
            const Quaternion turned = ptm::product(adjusted, turn);
            const double d = disorientation(hexagonal_q, turned, hexagonal_group());
            if (d < best) {
                closest = turned;
                best = d;
            }
        }
    }
    q = closest;
    ptm::into_fundamental_zone(hexagonal_group(), q);
    return best;
}

// Union of sets by size; find halves the paths.
class DisjointSet {
public:
    explicit DisjointSet(std::int64_t n) : parent_(static_cast<std::size_t>(n)), size_(static_cast<std::size_t>(n)) { clear(); }

    void clear() {
        std::iota(parent_.begin(), parent_.end(), std::int64_t{0});
        std::fill(size_.begin(), size_.end(), std::int64_t{1});
    }

    std::int64_t find(std::int64_t i) {
        std::int64_t x = parent_[i];
        while (x != parent_[x]) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        parent_[i] = x;
        return x;
    }

    std::int64_t merge(std::int64_t i, std::int64_t j) {
        const std::int64_t a = find(i), b = find(j);
        if (a == b) {
            return a;
        }
        if (size_[a] < size_[b]) {
            parent_[a] = b;
            size_[b] += size_[a];
            return b;
        }
        parent_[b] = a;
        size_[a] += size_[b];
        return a;
    }

    std::int64_t size(std::int64_t root) const { return size_[root]; }

private:
    std::vector<std::int64_t> parent_, size_;
};

// The weighted graph of node pair sampling: the weight of each node is the
// sum of those of its edges.
class Graph {
public:
    explicit Graph(std::int64_t n) : weight_(static_cast<std::size_t>(n), 0.0), edges_(static_cast<std::size_t>(n)) {}

    void add(std::int64_t u, std::int64_t v, double w) {
        for (std::int64_t node : {u, v}) {
            if (edges_[node].empty()) {
                active_.insert(node);
            }
            weight_[node] += w;
        }
        edges_[u][v] = w;
        edges_[v][u] = w;
    }

    bool empty() const { return active_.empty(); }
    std::int64_t first() const { return *active_.begin(); }
    void remove(std::int64_t u) { active_.erase(u); }

    // The neighbor b of a with the smallest weight(b) / w(a, b), the lowest
    // index among equals, and that ratio times weight(a); none: b = -1.
    std::pair<double, std::int64_t> nearest(std::int64_t a) const {
        double best = kMax;
        std::int64_t found = -1;
        for (const auto& [v, w] : edges_[a]) {
            const double d = weight_[v] / w;
            if (d < best) {
                best = d;
                found = v;
            } else if (d == best) {
                found = std::min(found, v);
            }
        }
        return {best * weight_[a], found};
    }

    // Merges b into a, or a into b when b has more edges: returns the node
    // kept.
    std::int64_t contract(std::int64_t a, std::int64_t b) {
        if (edges_[b].size() > edges_[a].size()) {
            std::swap(a, b);
        }
        edges_[a].erase(b);
        edges_[b].erase(a);
        for (const auto& [v, w] : edges_[b]) {
            edges_[v].erase(b);
            auto it = edges_[a].find(v);
            if (it != edges_[a].end()) {
                it->second += w;
                edges_[v][a] += w;
            } else {
                edges_[a][v] = w;
                edges_[v][a] = w;
            }
        }
        edges_[b].clear();
        remove(b);
        weight_[a] += weight_[b];
        return a;
    }

private:
    std::vector<double> weight_;
    std::vector<std::map<std::int64_t, double>> edges_;
    std::set<std::int64_t> active_;
};

// The weight of an edge for node pair sampling.
double graph_weight(double disorientation) {
    if (disorientation < 1e-5) {
        disorientation = 0;
    }
    return std::exp(-disorientation * disorientation / 3);
}

// Adds the orientation of a cluster to that of another, turned by the
// symmetry of the lattice to the nearest equivalent: the norms of the
// quaternions are the sizes of the clusters. Returns the misorientation in
// radians.
double accumulate(int type, Quaternion& qa, const Quaternion& qb) {
    const std::vector<Quaternion>* group = group_of(type);
    if (group == nullptr) {
        return kMax;
    }
    const double na = std::sqrt(qa[0] * qa[0] + qa[1] * qa[1] + qa[2] * qa[2] + qa[3] * qa[3]);
    const double nb = std::sqrt(qb[0] * qb[0] + qb[1] * qb[1] + qb[2] * qb[2] + qb[3] * qb[3]);
    const Quaternion target = {qa[0] / na, qa[1] / na, qa[2] / na, qa[3] / na};
    Quaternion q = {qb[0] / nb, qb[1] / nb, qb[2] / nb, qb[3] / nb};
    // the element g of the group that brings q g nearest the target
    Quaternion turn = ptm::product(Quaternion{-target[0], target[1], target[2], target[3]}, q);
    const int g = ptm::into_fundamental_zone(*group, turn);
    if (g < 0) {
        return kInfinity;
    }
    q = ptm::product(q, (*group)[static_cast<std::size_t>(g)]);
    if (q[0] < 0) {
        q = {-q[0], -q[1], -q[2], -q[3]};
    }
    const double t = std::clamp(q[0] * target[0] + q[1] * target[1] + q[2] * target[2] + q[3] * target[3], -1.0, 1.0);
    for (int k = 0; k < 4; ++k) {
        qa[static_cast<std::size_t>(k)] += q[static_cast<std::size_t>(k)] * nb;
    }
    return std::acos(2 * t * t - 1);
}

// The median, the mean of the two middle values for an even count.
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    double m = values[n / 2];
    if (n % 2 == 0) {
        m = (m + values[n / 2 - 1]) / 2;
    }
    return m;
}

void weighted_line(std::vector<double>& w, const std::vector<double>& x, const std::vector<double>& y,
                   double& slope, double& intercept) {
    double total = 0;
    for (double v : w) total += v;
    for (double& v : w) v /= total;
    double xm = 0, ym = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        xm += w[i] * x[i];
        ym += w[i] * y[i];
    }
    double sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        sxx += w[i] * (x[i] - xm) * (x[i] - xm);
        sxy += w[i] * (x[i] - xm) * (y[i] - ym);
    }
    slope = sxy / sxx;
    intercept = ym - slope * xm;
}

// The threshold of graph clustering: the line through log merge distance
// against log merge size by least absolute deviations (reweighted least
// squares), and the largest log distance of the merges that do not lie
// above it by more than 1.5 median deviations.
double suggest_threshold(const std::vector<Merge>& merges) {
    if (merges.empty()) {
        return 0.0;
    }
    std::vector<double> weights, x, y;
    for (const Merge& m : merges) {
        weights.push_back(m.merge_size);
        x.push_back(std::log(m.merge_size));
        y.push_back(std::log(m.distance));
    }
    std::vector<double> residuals(weights.size()), w(weights);
    double slope = 0, intercept = 0;
    for (int iteration = 0; iteration < 100; ++iteration) {
        weighted_line(w, x, y, slope, intercept);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double r = std::abs(y[i] - (slope * x[i] + intercept));
            residuals[i] = r;
            w[i] = weights[i] / std::max(1e-4, r);
        }
    }
    const double deviation = median(residuals);
    double threshold = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double residual = y[i] - (x[i] * slope + intercept);
        if (residual < 1.5 * deviation) {
            threshold = std::max(threshold, y[i]);
        }
    }
    return threshold;
}

// Disjoint sets that also keep the periodic image of each atom relative to
// the root of its set, to unwrap grains across the boundaries of the box.
class ImageSets {
public:
    explicit ImageSets(std::int64_t n)
        : parent_(static_cast<std::size_t>(n)), image_(static_cast<std::size_t>(n), {0, 0, 0}),
          rank_(static_cast<std::size_t>(n), 0) {
        std::iota(parent_.begin(), parent_.end(), std::int64_t{0});
    }

    std::pair<std::int64_t, std::array<long, 3>> find(std::int64_t i) {
        std::int64_t root = i;
        std::array<long, 3> image = {0, 0, 0};
        while (parent_[root] != root) {
            for (int k = 0; k < 3; ++k) image[k] += image_[root][k];
            root = parent_[root];
        }
        std::int64_t current = i;
        std::array<long, 3> current_image = image;
        while (parent_[current] != root) {
            const std::int64_t next = parent_[current];
            std::array<long, 3> next_image;
            for (int k = 0; k < 3; ++k) next_image[k] = current_image[k] - image_[current][k];
            parent_[current] = root;
            image_[current] = current_image;
            current = next;
            current_image = next_image;
        }
        return {root, image};
    }

    void merge(std::int64_t a, std::int64_t b, const std::array<long, 3>& image) {
        auto [ra, ia] = find(a);
        auto [rb, ib] = find(b);
        if (ra == rb) {
            return;
        }
        if (rank_[ra] < rank_[rb]) {
            parent_[ra] = rb;
            for (int k = 0; k < 3; ++k) image_[ra][k] = ib[k] - ia[k] - image[k];
        } else {
            parent_[rb] = ra;
            for (int k = 0; k < 3; ++k) image_[rb][k] = ia[k] - ib[k] + image[k];
            if (rank_[ra] == rank_[rb]) ++rank_[ra];
        }
    }

private:
    std::vector<std::int64_t> parent_;
    std::vector<std::array<long, 3>> image_;
    std::vector<std::uint8_t> rank_;
};

void invert(const double* m, double* inv) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    inv[0] = (m[4] * m[8] - m[5] * m[7]) / det;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) / det;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) / det;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) / det;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) / det;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) / det;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) / det;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) / det;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) / det;
}

}  // namespace

void grain_segmentation(const GrainInput& in, const GrainOptions& options, GrainOutput& out,
                        const Control& control) {
    const std::int64_t n = in.n;
    for (std::int64_t e = 0; e < in.offsets[n]; ++e) {
        if (in.neighbors[e] < 0 || in.neighbors[e] >= n) {
            throw std::invalid_argument("a neighbor is outside the atoms");
        }
    }
    auto neighbors_of = [&](std::int64_t i) {
        return std::pair<std::int64_t, std::int64_t>{in.offsets[i], in.offsets[i + 1]};
    };

    // the bonds between neighbors, once each: from the atom of lower index
    std::vector<Bond> bonds;
    for (std::int64_t i = 0; i < n; ++i) {
        auto [begin, end] = neighbors_of(i);
        if (in.structure[i] == kPtmOther) {
            end = std::min(end, begin + kDisorderedNeighbors);
        }
        for (std::int64_t e = begin; e < end; ++e) {
            const std::int64_t j = in.neighbors[e];
            if (i < j) {
                bonds.push_back({i, j, kInfinity, in.lengths[e]});
            }
        }
    }
    checkpoint(control, 1, 6);

    std::vector<std::int8_t> type(in.structure, in.structure + n);
    std::vector<Quaternion> orientation(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const double* q = in.orientation + 4 * i;
        orientation[i] = {q[3], q[0], q[1], q[2]};
    }

    // atoms of the minority phase at a coherent interface take the phase and
    // the orientation of the majority, spreading from the bonds of least
    // misorientation
    if (options.interfaces) {
        std::int64_t counts[9] = {0};
        for (std::int64_t i = 0; i < n; ++i) {
            if (type[i] >= 0 && type[i] < 9) ++counts[type[i]];
        }
        const bool parent_fcc = counts[kPtmFcc] >= counts[kPtmHcp];
        const bool parent_cubic_diamond = counts[kPtmCubicDiamond] >= counts[kPtmHexagonalDiamond];
        auto parent_phase = [&](int defect) -> std::int8_t {
            if (defect == kPtmHcp) return kPtmFcc;
            if (defect == kPtmFcc) return kPtmHcp;
            if (defect == kPtmHexagonalDiamond) return kPtmCubicDiamond;
            return kPtmHexagonalDiamond;
        };
        // orders a bond (majority atom, minority atom); false if it is not
        // across such an interface
        auto order = [&](Bond& bond) {
            const int sa = type[bond.a], sb = type[bond.b];
            bool flip;
            if (sa == kPtmFcc && sb == kPtmHcp) {
                flip = !parent_fcc;
            } else if (sa == kPtmHcp && sb == kPtmFcc) {
                flip = parent_fcc;
            } else if (sa == kPtmCubicDiamond && sb == kPtmHexagonalDiamond) {
                flip = !parent_cubic_diamond;
            } else if (sa == kPtmHexagonalDiamond && sb == kPtmCubicDiamond) {
                flip = parent_cubic_diamond;
            } else {
                return false;
            }
            if (flip) {
                std::swap(bond.a, bond.b);
            }
            return true;
        };
        // the orientation the minority atom would take, in turned
        auto across = [&](Bond& bond, Quaternion& turned) {
            bond.disorientation = kInfinity;
            if (!order(bond)) {
                return false;
            }
            turned = orientation[bond.b];
            const int sa = type[bond.a];
            const double d = (sa == kPtmFcc || sa == kPtmCubicDiamond)
                                 ? hexagonal_to_cubic(orientation[bond.a], turned)
                                 : cubic_to_hexagonal(orientation[bond.a], turned);
            bond.disorientation = d * kDegrees;
            return bond.disorientation < kMisorientationCutoff;
        };
        auto later = [](const Bond& x, const Bond& y) { return x.disorientation > y.disorientation; };
        std::priority_queue<Bond, std::vector<Bond>, decltype(later)> queue(later);
        Quaternion turned;
        for (Bond bond : bonds) {
            if (across(bond, turned)) {
                queue.push({bond.a, bond.b, bond.disorientation, 0.0});
            }
        }
        std::int64_t steps = 0;
        while (!queue.empty()) {
            Bond bond = queue.top();
            queue.pop();
            if (!across(bond, turned)) {
                continue;
            }
            const std::int64_t atom = bond.b;
            type[atom] = parent_phase(type[atom]);
            orientation[atom] = turned;
            auto [begin, end] = neighbors_of(atom);
            for (std::int64_t e = begin; e < end; ++e) {
                Bond next = {atom, in.neighbors[e], kInfinity, 0.0};
                if (across(next, turned)) {
                    queue.push({next.a, next.b, next.disorientation, 0.0});
                }
            }
            if (++steps % 16384 == 0) {
                checkpoint(control, 2, 6);
            }
        }
    }
    checkpoint(control, 2, 6);

    auto crystalline = [&](const Bond& bond) {
        const int a = type[bond.a], b = type[bond.b];
        if (a == kPtmOther || b == kPtmOther) return false;
        if (a == b) return true;
        if (!options.interfaces) return false;
        return (a == kPtmFcc && b == kPtmHcp) || (a == kPtmHcp && b == kPtmFcc) ||
               (a == kPtmCubicDiamond && b == kPtmHexagonalDiamond) ||
               (a == kPtmHexagonalDiamond && b == kPtmCubicDiamond);
    };
    parallel_for(
        static_cast<std::int64_t>(bonds.size()), 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t k = begin; k < end; ++k) {
                Bond& bond = bonds[static_cast<std::size_t>(k)];
                bond.disorientation =
                    bond_disorientation(type[bond.a], type[bond.b], orientation[bond.a], orientation[bond.b]);
            }
        },
        stage(control, 0, 1));
    std::sort(bonds.begin(), bonds.end(),
              [](const Bond& x, const Bond& y) { return x.disorientation < y.disorientation; });
    checkpoint(control, 3, 6);

    // the merges, cluster by cluster, with the orientations of the clusters
    std::vector<Quaternion> sum(orientation);
    std::vector<Merge> merges;
    if (options.spanning_tree) {
        DisjointSet sets(n);
        for (const Bond& bond : bonds) {
            if (bond.disorientation < kMisorientationCutoff) {
                const std::int64_t pa = sets.find(bond.a), pb = sets.find(bond.b);
                if (pa != pb && crystalline(bond)) {
                    const std::int64_t parent = sets.merge(pa, pb);
                    const std::int64_t child = parent == pa ? pb : pa;
                    const double d = accumulate(type[parent], sum[parent], sum[child]);
                    merges.push_back({parent, child, bond.disorientation, d, 1, 0.0, sum[parent]});
                }
            }
        }
    } else {
        Graph graph(n);
        for (const Bond& bond : bonds) {
            if (crystalline(bond) && bond.disorientation < kMisorientationCutoff) {
                graph.add(bond.a, bond.b, graph_weight(bond.disorientation));
            }
        }
        // nearest-neighbor chains
        std::vector<std::int64_t> chain;
        std::int64_t steps = 0;
        while (!graph.empty()) {
            chain.push_back(graph.first());
            while (!chain.empty()) {
                const std::int64_t a = chain.back();
                chain.pop_back();
                const auto [d, b] = graph.nearest(a);
                if (b < 0) {
                    graph.remove(a);  // a connected component is done
                } else if (!chain.empty()) {
                    const std::int64_t c = chain.back();
                    chain.pop_back();
                    if (b == c) {
                        const std::int64_t parent = graph.contract(a, b);
                        const std::int64_t child = parent == a ? b : a;
                        const double dis = accumulate(type[parent], sum[parent], sum[child]);
                        merges.push_back({parent, child, d, dis, 1, 0.0, sum[parent]});
                        if (++steps % 16384 == 0) {
                            checkpoint(control, 4, 6);
                        }
                    } else {
                        chain.push_back(c);
                        chain.push_back(a);
                        chain.push_back(b);
                    }
                } else {
                    chain.push_back(a);
                    chain.push_back(b);
                }
            }
        }
    }
    std::sort(merges.begin(), merges.end(), [](const Merge& x, const Merge& y) { return x.distance < y.distance; });
    checkpoint(control, 4, 6);

    // the sizes of the clusters each merge joins
    {
        DisjointSet sets(n);
        for (Merge& m : merges) {
            const double sa = static_cast<double>(sets.size(sets.find(m.a)));
            const double sb = static_cast<double>(sets.size(sets.find(m.b)));
            m.size = static_cast<std::int64_t>(std::min(sa, sb));
            m.merge_size = 2.0 / (1.0 / sa + 1.0 / sb);
            sets.merge(m.a, m.b);
            if (m.size >= kMinPlotSize) {
                out.merge_distance.push_back(m.distance);
                out.merge_size.push_back(static_cast<double>(m.size));
            }
        }
    }
    double threshold;
    if (options.spanning_tree) {
        threshold = std::log(options.threshold);
    } else {
        out.suggested = suggest_threshold(merges);
        threshold = options.automatic ? out.suggested : options.threshold;
    }
    out.threshold = options.spanning_tree ? options.threshold : threshold;

    // the merges up to the threshold
    std::vector<Quaternion> mean(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const double* q = in.orientation + 4 * i;
        mean[i] = {q[3], q[0], q[1], q[2]};
    }
    DisjointSet sets(n);
    for (const Merge& m : merges) {
        if (std::log(m.distance) > threshold) {
            break;
        }
        sets.merge(m.a, m.b);
        mean[sets.find(m.a)] = m.orientation;
    }

    // grains: clusters of crystalline atoms large enough, then by size
    std::vector<std::int64_t> label(static_cast<std::size_t>(n));
    std::vector<std::int8_t> grain_type;
    std::vector<Quaternion> grain_orientation;
    std::int64_t count = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if (sets.find(i) != i) {
            continue;
        }
        if (sets.size(i) < options.min_size || in.structure[i] == kPtmOther) {
            label[i] = -1;
        } else {
            label[i] = count++;
            grain_type.push_back(in.structure[i]);
            const Quaternion& q = mean[i];
            const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            grain_orientation.push_back({q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm});
        }
    }
    out.grain.assign(static_cast<std::size_t>(n), -1);
    std::vector<std::int64_t> size(static_cast<std::size_t>(count), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t g = label[sets.find(i)];
        out.grain[i] = g;
        if (g >= 0) ++size[g];
    }
    std::vector<std::int64_t> order(static_cast<std::size_t>(count));
    std::iota(order.begin(), order.end(), std::int64_t{0});
    std::sort(order.begin(), order.end(), [&](std::int64_t x, std::int64_t y) { return size[x] > size[y]; });
    std::vector<std::int64_t> rank(static_cast<std::size_t>(count));
    for (std::int64_t k = 0; k < count; ++k) rank[order[k]] = k;
    for (std::int64_t& g : out.grain) {
        if (g >= 0) g = rank[g];
    }
    out.size.resize(static_cast<std::size_t>(count));
    out.structure.resize(static_cast<std::size_t>(count));
    out.orientation.resize(static_cast<std::size_t>(4 * count));
    for (std::int64_t k = 0; k < count; ++k) {
        out.size[k] = size[order[k]];
        out.structure[k] = grain_type[order[k]];
        const Quaternion& q = grain_orientation[order[k]];
        out.orientation[4 * k] = q[1];
        out.orientation[4 * k + 1] = q[2];
        out.orientation[4 * k + 2] = q[3];
        out.orientation[4 * k + 3] = q[0];
    }
    checkpoint(control, 5, 6);

    // atoms in no grain join the grain nearest along the bonds
    if (options.adopt_orphans && count > 0) {
        std::vector<Bond> loose;
        for (const Bond& bond : bonds) {
            if (out.grain[bond.a] < 0 || out.grain[bond.b] < 0) {
                loose.push_back(bond);
                loose.push_back({bond.b, bond.a, bond.disorientation, bond.length});
            }
        }
        std::sort(loose.begin(), loose.end(), [](const Bond& x, const Bond& y) { return x.a < y.a; });
        struct Step {
            std::int64_t grain, atom;
            double length;
        };
        auto later = [](const Step& x, const Step& y) { return x.length > y.length; };
        std::priority_queue<Step, std::vector<Step>, decltype(later)> queue(later);
        for (const Bond& bond : bonds) {
            const std::int64_t ga = out.grain[bond.a], gb = out.grain[bond.b];
            if (ga >= 0 && gb < 0) {
                queue.push({ga, bond.b, bond.length});
            } else if (ga < 0 && gb >= 0) {
                queue.push({gb, bond.a, bond.length});
            }
        }
        while (!queue.empty()) {
            const Step step = queue.top();
            queue.pop();
            if (out.grain[step.atom] >= 0) {
                continue;
            }
            out.grain[step.atom] = step.grain;
            ++out.size[step.grain];
            auto range = std::equal_range(loose.begin(), loose.end(), Bond{step.atom, 0, 0.0, 0.0},
                                          [](const Bond& x, const Bond& y) { return x.a < y.a; });
            for (auto it = range.first; it != range.second; ++it) {
                if (out.grain[it->b] >= 0) {
                    continue;
                }
                queue.push({step.grain, it->b, step.length + it->length});
            }
        }
    }

    // centers, radii and radii of gyration, the grains unwrapped across the
    // boundaries of the box along their bonds
    out.center.assign(static_cast<std::size_t>(3 * count), 0.0);
    out.radius.assign(static_cast<std::size_t>(count), 0.0);
    out.gyration.assign(static_cast<std::size_t>(count), 0.0);
    if (count == 0 || in.positions == nullptr) {
        return;
    }
    double inverse[9];
    invert(in.matrix, inverse);
    const double* x = in.positions;
    auto fractional = [&](const double* d, int k) {
        return d[0] * inverse[k] + d[1] * inverse[3 + k] + d[2] * inverse[6 + k];
    };
    ImageSets images(n);
    for (const Bond& bond : bonds) {
        const std::int64_t g = out.grain[bond.a];
        if (g < 0 || g != out.grain[bond.b]) {
            continue;
        }
        const double d[3] = {x[3 * bond.b] - x[3 * bond.a], x[3 * bond.b + 1] - x[3 * bond.a + 1],
                             x[3 * bond.b + 2] - x[3 * bond.a + 2]};
        std::array<long, 3> image = {0, 0, 0};
        for (int k = 0; k < 3; ++k) {
            if (in.pbc[k]) image[k] = -static_cast<long>(std::floor(fractional(d, k) + 0.5));
        }
        images.merge(bond.a, bond.b, image);
    }
    auto unwrapped = [&](std::int64_t i, double* p) {
        const auto image = images.find(i).second;
        for (int k = 0; k < 3; ++k) {
            p[k] = x[3 * i + k] + image[0] * in.matrix[k] + image[1] * in.matrix[3 + k] + image[2] * in.matrix[6 + k];
        }
    };
    std::vector<double> weighted(static_cast<std::size_t>(3 * count), 0.0), total(static_cast<std::size_t>(count), 0.0);
    std::vector<double> plain(static_cast<std::size_t>(3 * count), 0.0);
    std::vector<std::int64_t> members(static_cast<std::size_t>(count), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t g = out.grain[i];
        if (g < 0) continue;
        double p[3];
        unwrapped(i, p);
        const double w = in.masses != nullptr ? in.masses[i] : 1.0;
        for (int k = 0; k < 3; ++k) {
            weighted[3 * g + k] += w * p[k];
            plain[3 * g + k] += p[k];
        }
        total[g] += w;
        ++members[g];
    }
    std::vector<double> centers(static_cast<std::size_t>(3 * count), 0.0);
    for (std::int64_t g = 0; g < count; ++g) {
        for (int k = 0; k < 3; ++k) {
            if (total[g] > 0) {
                centers[3 * g + k] = weighted[3 * g + k] / total[g];
            } else if (members[g] > 0) {
                centers[3 * g + k] = plain[3 * g + k] / static_cast<double>(members[g]);
            }
        }
        // the center wrapped into the box
        double d[3] = {centers[3 * g] - in.origin[0], centers[3 * g + 1] - in.origin[1],
                       centers[3 * g + 2] - in.origin[2]};
        double f[3];
        for (int k = 0; k < 3; ++k) {
            f[k] = fractional(d, k);
            if (in.pbc[k]) f[k] -= std::floor(f[k]);
        }
        for (int k = 0; k < 3; ++k) {
            out.center[3 * g + k] =
                in.origin[k] + f[0] * in.matrix[k] + f[1] * in.matrix[3 + k] + f[2] * in.matrix[6 + k];
        }
    }
    std::vector<double> weighted_sq(static_cast<std::size_t>(count), 0.0), plain_sq(static_cast<std::size_t>(count), 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t g = out.grain[i];
        if (g < 0) continue;
        double p[3];
        unwrapped(i, p);
        double r2 = 0;
        for (int k = 0; k < 3; ++k) {
            const double d = p[k] - centers[3 * g + k];
            r2 += d * d;
        }
        const double w = in.masses != nullptr ? in.masses[i] : 1.0;
        weighted_sq[g] += w * r2;
        plain_sq[g] += r2;
        out.radius[g] = std::max(out.radius[g], r2);
    }
    for (std::int64_t g = 0; g < count; ++g) {
        out.radius[g] = std::sqrt(out.radius[g]);
        if (total[g] > 0) {
            out.gyration[g] = std::sqrt(weighted_sq[g] / total[g]);
        } else if (members[g] > 0) {
            out.gyration[g] = std::sqrt(plain_sq[g] / static_cast<double>(members[g]));
        }
    }
    checkpoint(control, 6, 6);
}

}  // namespace mdapy::identify

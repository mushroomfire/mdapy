// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// The tree follows the k-d tree of mdapy 1.x: it is implicit, with no node
// records. The points of a node are a range of the array with its median
// point in the middle, the points below it across the axis of the node's
// depth before and those above after. A search goes to the side of the
// probe first, and to the other side when the cell there, whose distance is
// kept up to date one axis at a time, is within the limit. Periodic images
// are visited by moving the probe.

#include "neighbor/kd_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace mdapy::neighbor {
namespace {

// Most points in a leaf.
constexpr std::int64_t kLeaf = 16;

// Fewest points split by a chunk of a parallel region.
constexpr std::int64_t kChunkPoints = 4096;

constexpr double kInf = std::numeric_limits<double>::infinity();

// The order of neighbors: by distance, then atom, then image.
inline bool before(const Neighbor& p, const Neighbor& q) {
    if (p.r2 != q.r2) {
        return p.r2 < q.r2;
    }
    if (p.j != q.j) {
        return p.j < q.j;
    }
    for (int e = 0; e < 3; ++e) {
        if (p.shift[e] != q.shift[e]) {
            return p.shift[e] < q.shift[e];
        }
    }
    return false;
}

inline bool nearer(const KdTree::Candidate& p, const KdTree::Candidate& q) { return p.r2 < q.r2; }

// A node: its points [begin, begin + count) and its depth.
struct Range {
    std::int64_t begin, count;
    int depth;
};

}  // namespace

// A query point moved by a lattice vector: the image q = y - image . H of
// its image y in the box, and what turns a point of the tree into an exact
// pair: the original position x and image + w, w its wraps.
struct KdTree::Probe {
    const double* x;
    double q[3];
    std::int32_t base[3];
    std::int64_t self;
};

// The candidates found so far, with their squared distances computed from
// the images in the box: the k nearest as a heap whose top is the k-th,
// then others as near as the k-th within twice the margin. Points and cells
// beyond the limit, the distance of the k-th plus twice the margin, cannot
// be among the k nearest by exact distance; the margin is far above the
// rounding of either distance.
class KdTree::Search {
public:
    Search(int k, std::vector<Candidate>& candidates) : k_(k), candidates_(candidates) {
        candidates_.clear();
    }

    bool full() const { return static_cast<int>(candidates_.size()) >= k_; }
    double limit2() const { return limit2_; }

    void set_margin(double margin) {
        margin_ = std::max(margin_, margin);
        update();
    }

    // c, within the limit.
    void push(const Candidate& c) {
        if (!full()) {
            candidates_.push_back(c);
            if (full()) {
                std::make_heap(candidates_.begin(), candidates_.end(),
                               [](const Candidate& p, const Candidate& q) { return nearer(p, q); });
                update();
            }
            return;
        }
        if (!(c.r2 < candidates_[0].r2)) {
            candidates_.push_back(c);
            return;
        }
        // c replaces the top and goes down, below its later children
        const Candidate out = candidates_[0];
        int at = 0;
        for (int child = 1; child < k_; child = 2 * at + 1) {
            if (child + 1 < k_ && nearer(candidates_[child], candidates_[child + 1])) {
                ++child;
            }
            if (!nearer(c, candidates_[child])) {
                break;
            }
            candidates_[at] = candidates_[child];
            at = child;
        }
        candidates_[at] = c;
        update();
        if (out.r2 <= limit2_) {
            candidates_.push_back(out);
        }
    }

    // Calls each(c) for the candidates still within the limit.
    template <class Each>
    void finish(Each&& each) const {
        for (const Candidate& c : candidates_) {
            if (c.r2 <= limit2_) {
                each(c);
            }
        }
    }

private:
    void update() {
        if (full()) {
            const double limit = std::sqrt(candidates_[0].r2) + 2.0 * margin_;
            limit2_ = limit * limit;
        }
    }

    int k_;
    std::vector<Candidate>& candidates_;
    double margin_ = 0.0;
    double limit2_ = kInf;
};

KdTree::KdTree(const double* positions, std::int64_t n, const double* matrix,
               const double* origin, const bool* pbc)
    : frame_(matrix, origin, pbc), periodic_(pbc[0] || pbc[1] || pbc[2]) {
    check_atom_count(n);
    std::vector<std::int32_t> wraps(static_cast<std::size_t>(n) * 3);
    points_.resize(static_cast<std::size_t>(n));
    struct Bounds {
        double low[3], high[3], size;
    };
    const Bounds none{{kInf, kInf, kInf}, {-kInf, -kInf, -kInf}, 0.0};
    const Bounds bounds = parallel_reduce(
        n, 0, none,
        [&](std::int64_t begin, std::int64_t end) {
            Bounds part = none;
            for (std::int64_t i = begin; i < end; ++i) {
                const double* x = positions + 3 * i;
                double f[3];
                Point& point = points_[i];
                frame_.wrap(x, f, &wraps[3 * i]);
                frame_.image(x, &wraps[3 * i], point.y);
                point.atom = static_cast<index_t>(i);
                for (int e = 0; e < 3; ++e) {
                    part.low[e] = std::min(part.low[e], point.y[e]);
                    part.high[e] = std::max(part.high[e], point.y[e]);
                    part.size = std::max(part.size, std::max(std::abs(x[e]), std::abs(point.y[e])));
                }
            }
            return part;
        },
        [](Bounds p, const Bounds& q) {
            for (int e = 0; e < 3; ++e) {
                p.low[e] = std::min(p.low[e], q.low[e]);
                p.high[e] = std::max(p.high[e], q.high[e]);
            }
            p.size = std::max(p.size, q.size);
            return p;
        });
    std::copy(bounds.low, bounds.low + 3, low_);
    std::copy(bounds.high, bounds.high + 3, high_);
    size_ = bounds.size;

    // Level by level, the median point of each node in its middle.
    std::vector<Range> level, next;
    if (n > kLeaf) {
        level.push_back({0, n, 0});
    }
    while (!level.empty()) {
        const auto count = static_cast<std::int64_t>(level.size());
        const std::int64_t grain = std::max<std::int64_t>(1, kChunkPoints * count / n);
        parallel_for(count, grain, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t a = begin; a < end; ++a) {
                const Range& node = level[a];
                const int axis = node.depth % 3;
                const auto first = points_.begin() + node.begin;
                std::nth_element(first, first + node.count / 2, first + node.count,
                                 [axis](const Point& p, const Point& q) { return p.y[axis] < q.y[axis]; });
            }
        });
        next.clear();
        for (const Range& node : level) {
            const std::int64_t below = node.count / 2, above = node.count - below - 1;
            for (const Range& child : {Range{node.begin, below, node.depth + 1},
                                       Range{node.begin + below + 1, above, node.depth + 1}}) {
                if (child.count > kLeaf) {
                    next.push_back(child);
                }
            }
        }
        level.swap(next);
    }

    positions_.resize(static_cast<std::size_t>(n) * 3);
    wraps_.resize(static_cast<std::size_t>(n) * 3);
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t t = begin; t < end; ++t) {
            const auto i = static_cast<std::int64_t>(points_[t].atom);
            for (int e = 0; e < 3; ++e) {
                positions_[3 * t + e] = positions[3 * i + e];
                wraps_[3 * t + e] = wraps[3 * i + e];
            }
        }
    });
}

void KdTree::check_count(int k, std::int64_t available) const {
    if (k < 1) {
        throw std::invalid_argument("the number of neighbors must be at least 1");
    }
    if (size() == 0) {
        throw std::invalid_argument("there are no atoms");
    }
    if (!periodic_ && available < k) {
        throw std::invalid_argument("a system without periodic directions has only " +
                                    std::to_string(available) + " neighbors for each point, not " +
                                    std::to_string(k));
    }
}

double KdTree::lattice_reach(int k) const {
    if (!periodic_) {
        return 0.0;
    }
    // The lengths of the lattice vectors in a block of images, enlarged
    // until a vector outside the block, at least (m + 1) t long along a
    // direction of thickness t where the block ends at m, cannot be shorter
    // than the k-th.
    const double* h = frame_.h;
    std::int64_t m[3];
    for (int d = 0; d < 3; ++d) {
        m[d] = frame_.pbc[d] ? 1 : 0;
    }
    std::vector<double> lengths;
    while (true) {
        lengths.clear();
        for (std::int64_t a = -m[0]; a <= m[0]; ++a) {
            for (std::int64_t b = -m[1]; b <= m[1]; ++b) {
                for (std::int64_t c = -m[2]; c <= m[2]; ++c) {
                    if (a == 0 && b == 0 && c == 0) {
                        continue;
                    }
                    double r2 = 0.0;
                    for (int e = 0; e < 3; ++e) {
                        const double r = static_cast<double>(a) * h[e] +
                                         static_cast<double>(b) * h[3 + e] +
                                         static_cast<double>(c) * h[6 + e];
                        r2 += r * r;
                    }
                    lengths.push_back(std::sqrt(r2));
                }
            }
        }
        if (static_cast<std::int64_t>(lengths.size()) < k) {
            for (int d = 0; d < 3; ++d) {
                m[d] *= 2;
            }
            continue;
        }
        std::nth_element(lengths.begin(), lengths.begin() + (k - 1), lengths.end());
        const double reach = lengths[static_cast<std::size_t>(k - 1)];
        bool done = true;
        for (int d = 0; d < 3; ++d) {
            if (frame_.pbc[d] && static_cast<double>(m[d] + 1) * frame_.thickness[d] <= reach) {
                m[d] = static_cast<std::int64_t>(std::ceil(reach / frame_.thickness[d]));
                done = false;
            }
        }
        if (done) {
            return reach;
        }
    }
}

void KdTree::scan(std::int64_t begin, std::int64_t end, const Probe& probe,
                  Search& search) const {
    const double* q = probe.q;
    for (std::int64_t t = begin; t < end; ++t) {
        const Point& point = points_[t];
        const double dx = point.y[0] - q[0], dy = point.y[1] - q[1], dz = point.y[2] - q[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > search.limit2()) {
            continue;
        }
        Candidate c{r2, t, {}};
        const std::int32_t* w = &wraps_[3 * t];
        for (int e = 0; e < 3; ++e) {
            c.shift[e] = probe.base[e] - w[e];
        }
        if (point.atom == probe.self && c.shift[0] == 0 && c.shift[1] == 0 && c.shift[2] == 0) {
            continue;
        }
        search.push(c);
    }
}

void KdTree::descend(std::int64_t begin, std::int64_t count, int depth, double* gap,
                     double distance2, const Probe& probe, Search& search) const {
    if (count <= kLeaf) {
        scan(begin, begin + count, probe, search);
        return;
    }
    const int axis = depth % 3;
    const std::int64_t below = count / 2, mid = begin + below;
    const double side = probe.q[axis] - points_[mid].y[axis];
    if (side < 0.0) {
        descend(begin, below, depth + 1, gap, distance2, probe, search);
    } else {
        descend(mid + 1, count - below - 1, depth + 1, gap, distance2, probe, search);
    }
    // The cell on the other side differs along this axis only, where it is
    // as far as the median point's plane; that point is in it.
    const double old = gap[axis];
    const double far2 = distance2 - old * old + side * side;
    if (far2 > search.limit2()) {
        return;
    }
    scan(mid, mid + 1, probe, search);
    gap[axis] = side;
    if (side < 0.0) {
        descend(mid + 1, count - below - 1, depth + 1, gap, far2, probe, search);
    } else {
        descend(begin, below, depth + 1, gap, far2, probe, search);
    }
    gap[axis] = old;
}

void KdTree::search(const double* x, std::int64_t self, int k, double reach,
                    std::vector<Candidate>& candidates, std::vector<Neighbor>& found) const {
    const double* h = frame_.h;
    double f[3], y[3];
    std::int32_t w[3];
    frame_.wrap(x, f, w);
    frame_.image(x, w, y);
    double scale = size_;
    for (int e = 0; e < 3; ++e) {
        scale = std::max(scale, std::max(std::abs(x[e]), std::abs(y[e])));
    }
    Search found_so_far(k, candidates);
    const auto visit = [&](const std::int64_t* image) {
        Probe probe;
        probe.x = x;
        probe.self = self;
        double extent = 4.0 * scale;
        for (int d = 0; d < 3; ++d) {
            probe.base[d] = static_cast<std::int32_t>(image[d]) + w[d];
            extent += static_cast<double>(std::abs(image[d])) * frame_.lengths[d];
        }
        for (int e = 0; e < 3; ++e) {
            probe.q[e] = y[e] - (static_cast<double>(image[0]) * h[e] +
                                 static_cast<double>(image[1]) * h[3 + e] +
                                 static_cast<double>(image[2]) * h[6 + e]);
        }
        found_so_far.set_margin(1e-12 * extent);
        // the gaps between the probe and the box of all points, one per axis
        double gap[3], distance2 = 0.0;
        for (int e = 0; e < 3; ++e) {
            gap[e] = std::max(std::max(low_[e] - probe.q[e], probe.q[e] - high_[e]), 0.0);
            distance2 += gap[e] * gap[e];
        }
        if (distance2 <= found_so_far.limit2()) {
            descend(0, size(), 0, gap, distance2, probe, found_so_far);
        }
    };
    // the k nearest candidates by exact distance, nearest first
    const auto finish = [&]() {
        found.clear();
        found_so_far.finish([&](const Candidate& c) {
            Neighbor p{0.0, points_[c.t].atom, {c.shift[0], c.shift[1], c.shift[2]}};
            double r[3];
            p.r2 = pair_vector(x, &positions_[3 * c.t], p.shift, h, r);
            found.push_back(p);
        });
        if (static_cast<int>(found.size()) < k) {
            throw std::logic_error("the search found fewer than k neighbors");
        }
        std::partial_sort(found.begin(), found.begin() + k, found.end(),
                          [](const Neighbor& p, const Neighbor& q) { return before(p, q); });
    };
    const std::int64_t zero[3] = {0, 0, 0};
    visit(zero);
    if (!periodic_) {
        finish();
        return;
    }
    // Other images whose points can be within the radius: along a periodic
    // direction of thickness t the fractional coordinates of the points are
    // in [0, 1], so image s needs |f + s - f_x| t <= radius for some f there.
    // Until k neighbors are found, the radius is that of the farthest point
    // of the tree plus the reach.
    const auto radius = [&]() {
        if (found_so_far.full()) {
            return std::sqrt(found_so_far.limit2());
        }
        double far2 = 0.0;
        for (int e = 0; e < 3; ++e) {
            const double g = std::max(std::abs(y[e] - low_[e]), std::abs(high_[e] - y[e]));
            far2 += g * g;
        }
        return std::sqrt(far2) + reach + 1e-12 * 4.0 * scale;
    };
    const double slack = 1e-9 + 1e-12 * 4.0 * scale /
                                    std::min({frame_.thickness[0], frame_.thickness[1],
                                              frame_.thickness[2]});
    const auto range = [&](int d, double r, std::int64_t& low, std::int64_t& high) {
        if (!frame_.pbc[d]) {
            low = high = 0;
            return;
        }
        const double span = r / frame_.thickness[d] + slack;
        low = static_cast<std::int64_t>(std::ceil(f[d] - 1.0 - span));
        high = static_cast<std::int64_t>(std::floor(f[d] + span));
    };
    std::int64_t low[3], high[3];
    const double r0 = radius();
    for (int d = 0; d < 3; ++d) {
        range(d, r0, low[d], high[d]);
    }
    for (std::int64_t a = low[0]; a <= high[0]; ++a) {
        for (std::int64_t b = low[1]; b <= high[1]; ++b) {
            for (std::int64_t c = low[2]; c <= high[2]; ++c) {
                const std::int64_t image[3] = {a, b, c};
                if (a == 0 && b == 0 && c == 0) {
                    continue;
                }
                bool needed = true;
                if (found_so_far.full()) {  // the radius shrinks as neighbors are found
                    const double r = radius();
                    for (int d = 0; d < 3 && needed; ++d) {
                        std::int64_t lo, hi;
                        range(d, r, lo, hi);
                        needed = lo <= image[d] && image[d] <= hi;
                    }
                }
                if (needed) {
                    visit(image);
                }
            }
        }
    }
    finish();
}

NeighborPairs KdTree::nearest_atoms(int k, const Control& control) const {
    const std::int64_t n = size();
    NeighborPairs result;
    result.offsets.resize(static_cast<std::size_t>(n) + 1);
    if (n == 0) {
        result.offsets[0] = 0;
        return result;
    }
    check_count(k, n - 1);
    for (std::int64_t i = 0; i <= n; ++i) {
        result.offsets[i] = i * k;
    }
    const auto total = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    result.neighbors.resize(total);
    result.shifts.resize(3 * total);
    result.distances.resize(total);
    const double reach = lattice_reach(k);
    // the atoms in the order of the tree, near ones one after the other
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<Candidate> candidates;
            std::vector<Neighbor> found;
            for (std::int64_t t = begin; t < end; ++t) {
                const auto i = static_cast<std::int64_t>(points_[t].atom);
                search(&positions_[3 * t], i, k, reach, candidates, found);
                for (int a = 0; a < k; ++a) {
                    const Neighbor& p = found[a];
                    const auto at = static_cast<std::size_t>(i * k + a);
                    check_shift(p.shift);
                    result.neighbors[at] = p.j;
                    for (int e = 0; e < 3; ++e) {
                        result.shifts[3 * at + e] = static_cast<std::int8_t>(p.shift[e]);
                    }
                    result.distances[at] = std::sqrt(p.r2);
                }
            }
        },
        control);
    return result;
}

void KdTree::nearest_points(const double* points, std::int64_t m, int k, index_t* j,
                            std::int8_t* shifts, double* distances,
                            const Control& control) const {
    check_count(k, size());
    const double reach = lattice_reach(k);
    parallel_for(
        m, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<Candidate> candidates;
            std::vector<Neighbor> found;
            for (std::int64_t p = begin; p < end; ++p) {
                search(points + 3 * p, -1, k, reach, candidates, found);
                for (int a = 0; a < k; ++a) {
                    const auto at = static_cast<std::size_t>(p * k + a);
                    check_shift(found[a].shift);
                    j[at] = found[a].j;
                    for (int e = 0; e < 3; ++e) {
                        shifts[3 * at + e] = static_cast<std::int8_t>(found[a].shift[e]);
                    }
                    distances[at] = std::sqrt(found[a].r2);
                }
            }
        },
        control);
}

NeighborPairs nearest(const double* positions, std::int64_t n, const double* matrix,
                      const double* origin, const bool* pbc, int k, const Control& control) {
    return KdTree(positions, n, matrix, origin, pbc).nearest_atoms(k, control);
}

}  // namespace mdapy::neighbor

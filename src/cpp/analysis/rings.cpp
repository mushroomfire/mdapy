// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/rings.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace mdapy::analysis {

namespace {

// An atom in one periodic image, relative to the image of the root.
struct Node {
    std::int64_t atom;
    std::array<std::int32_t, 3> shift;
    int depth;
    std::vector<int> parents;  // nodes one step nearer the root
};

std::int64_t key(std::int64_t atom, const std::array<std::int32_t, 3>& s) {
    constexpr std::int64_t kSide = 1024;
    return ((atom * kSide + (s[0] + kSide / 2)) * kSide + (s[1] + kSide / 2)) * kSide +
           (s[2] + kSide / 2);
}

// The atoms within a number of bonds of an atom in some image, by breadth:
// every shortest path from it, as the parents of each node.
class Ball {
public:
    Ball(const RingBonds& bonds, std::int64_t atom, const std::array<std::int32_t, 3>& shift,
         int depth) {
        add(atom, shift, 0, -1);
        for (std::size_t k = 0; k < nodes_.size(); ++k) {
            const Node u = nodes_[k];  // a copy: nodes_ grows
            if (u.depth == depth) {
                continue;
            }
            for (std::int64_t b = bonds.offsets[u.atom]; b < bonds.offsets[u.atom + 1]; ++b) {
                const std::array<std::int32_t, 3> s = {u.shift[0] + bonds.shifts[3 * b],
                                                       u.shift[1] + bonds.shifts[3 * b + 1],
                                                       u.shift[2] + bonds.shifts[3 * b + 2]};
                add(bonds.other[b], s, u.depth + 1, static_cast<int>(k));
            }
        }
    }

    const std::vector<Node>& nodes() const { return nodes_; }

    // the node of an atom in an image, -1 where farther than the depth
    int find(std::int64_t atom, const std::array<std::int32_t, 3>& shift) const {
        const auto found = index_.find(key(atom, shift));
        return found == index_.end() ? -1 : found->second;
    }

private:
    void add(std::int64_t atom, const std::array<std::int32_t, 3>& shift, int depth, int parent) {
        const std::int64_t k = key(atom, shift);
        const auto found = index_.find(k);
        if (found == index_.end()) {
            index_.emplace(k, static_cast<int>(nodes_.size()));
            nodes_.push_back({atom, shift, depth, {}});
            if (parent >= 0) {
                nodes_.back().parents.push_back(parent);
            }
        } else if (nodes_[static_cast<std::size_t>(found->second)].depth == depth && parent >= 0) {
            nodes_[static_cast<std::size_t>(found->second)].parents.push_back(parent);
        }
    }

    std::vector<Node> nodes_;
    std::unordered_map<std::int64_t, int> index_;
};

// The shortest paths from the root to a node through atoms of larger index
// than the root's, each from the root.
class Paths {
public:
    explicit Paths(const Ball& ball) : ball_(ball), found_(ball.nodes().size()), made_(ball.nodes().size(), false) {}

    const std::vector<std::vector<int>>& to(int node) {
        const auto k = static_cast<std::size_t>(node);
        if (made_[k]) {
            return found_[k];
        }
        made_[k] = true;
        const Node& n = ball_.nodes()[k];
        if (n.depth == 0) {
            found_[k] = {{node}};
            return found_[k];
        }
        if (n.atom <= ball_.nodes()[0].atom) {
            return found_[k];
        }
        std::vector<std::vector<int>> paths;
        for (const int parent : n.parents) {
            for (const auto& path : to(parent)) {
                paths.push_back(path);
                paths.back().push_back(node);
            }
        }
        found_[k] = std::move(paths);
        return found_[k];
    }

private:
    const Ball& ball_;
    std::vector<std::vector<std::vector<int>>> found_;
    std::vector<bool> made_;
};

// Whether two atoms of a ring are closer through the network than along it.
bool shortcut(const RingBonds& bonds, const Ball& ball, const std::vector<int>& ring) {
    const auto n = static_cast<int>(ring.size());
    for (int i = 1; i < n; ++i) {
        const Node& from = ball.nodes()[static_cast<std::size_t>(ring[static_cast<std::size_t>(i)])];
        const Ball near(bonds, from.atom, from.shift, n / 2 - 1);
        for (int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const Node& to = ball.nodes()[static_cast<std::size_t>(ring[static_cast<std::size_t>(j)])];
            const int found = near.find(to.atom, to.shift);
            const int along = std::min(std::abs(i - j), n - std::abs(i - j));
            if (found >= 0 && near.nodes()[static_cast<std::size_t>(found)].depth < along) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

Rings find_rings(const RingBonds& bonds, int min_size, int max_size, const Control& control) {
    const std::int64_t n = bonds.n;
    // the rings of each root: atoms, then images
    std::vector<std::vector<std::int64_t>> atoms(static_cast<std::size_t>(n));
    std::vector<std::vector<std::int32_t>> images(static_cast<std::size_t>(n));
    std::vector<std::vector<std::int64_t>> sizes(static_cast<std::size_t>(n));
    parallel_for(
        n, 1,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t root = begin; root < end; ++root) {
                const Ball ball(bonds, root, {0, 0, 0}, max_size / 2);
                Paths paths(ball);
                const auto& nodes = ball.nodes();
                // each ring as its nodes from the root
                std::vector<std::vector<int>> found;
                auto try_ring = [&](const std::vector<int>& p, const std::vector<int>& q,
                                    bool shared_end) {
                    std::unordered_set<int> seen(p.begin() + 1, p.end());
                    const std::size_t last = shared_end ? q.size() - 1 : q.size();
                    for (std::size_t k = 1; k < last; ++k) {
                        if (seen.count(q[k])) {
                            return;
                        }
                    }
                    std::vector<int> ring(p.begin(), p.end());
                    for (std::size_t k = last; k-- > 1;) {
                        ring.push_back(q[k]);
                    }
                    const auto size = static_cast<int>(ring.size());
                    if (size < min_size || size > max_size || shortcut(bonds, ball, ring)) {
                        return;
                    }
                    // on toward the smaller of the root's two neighbors
                    auto order = [&](int u) {
                        const Node& x = nodes[static_cast<std::size_t>(u)];
                        return std::make_tuple(x.atom, x.shift[0], x.shift[1], x.shift[2]);
                    };
                    if (order(ring.back()) < order(ring[1])) {
                        std::reverse(ring.begin() + 1, ring.end());
                    }
                    found.push_back(std::move(ring));
                };
                for (std::size_t a = 1; a < nodes.size(); ++a) {
                    const int depth = nodes[a].depth;
                    const auto& to_a = paths.to(static_cast<int>(a));
                    // even rings: two paths to a node opposite the root
                    if (2 * depth >= min_size && 2 * depth <= max_size) {
                        for (std::size_t i = 0; i < to_a.size(); ++i) {
                            for (std::size_t j = i + 1; j < to_a.size(); ++j) {
                                try_ring(to_a[i], to_a[j], true);
                            }
                        }
                    }
                    // odd rings: paths to the two ends of a bond opposite it
                    if (2 * depth + 1 > max_size || 2 * depth + 1 < min_size) {
                        continue;
                    }
                    const Node& u = nodes[a];
                    for (std::int64_t b = bonds.offsets[u.atom]; b < bonds.offsets[u.atom + 1]; ++b) {
                        const std::array<std::int32_t, 3> s = {u.shift[0] + bonds.shifts[3 * b],
                                                               u.shift[1] + bonds.shifts[3 * b + 1],
                                                               u.shift[2] + bonds.shifts[3 * b + 2]};
                        const int c = ball.find(bonds.other[b], s);
                        if (c <= static_cast<int>(a) || nodes[static_cast<std::size_t>(c)].depth != depth) {
                            continue;
                        }
                        const auto& to_c = paths.to(c);
                        for (const auto& p : to_a) {
                            for (const auto& q : to_c) {
                                try_ring(p, q, false);
                            }
                        }
                    }
                }
                std::sort(found.begin(), found.end(), [&](const std::vector<int>& x, const std::vector<int>& y) {
                    if (x.size() != y.size()) {
                        return x.size() < y.size();
                    }
                    for (std::size_t k = 0; k < x.size(); ++k) {
                        const Node& p = nodes[static_cast<std::size_t>(x[k])];
                        const Node& q = nodes[static_cast<std::size_t>(y[k])];
                        if (p.atom != q.atom) {
                            return p.atom < q.atom;
                        }
                        if (p.shift != q.shift) {
                            return p.shift < q.shift;
                        }
                    }
                    return false;
                });
                auto& my_atoms = atoms[static_cast<std::size_t>(root)];
                auto& my_images = images[static_cast<std::size_t>(root)];
                auto& my_sizes = sizes[static_cast<std::size_t>(root)];
                for (const auto& ring : found) {
                    my_sizes.push_back(static_cast<std::int64_t>(ring.size()));
                    for (const int u : ring) {
                        const Node& x = nodes[static_cast<std::size_t>(u)];
                        my_atoms.push_back(x.atom);
                        my_images.insert(my_images.end(), x.shift.begin(), x.shift.end());
                    }
                }
            }
        },
        control);
    Rings out;
    out.starts.push_back(0);
    for (std::size_t r = 0; r < static_cast<std::size_t>(n); ++r) {
        out.atoms.insert(out.atoms.end(), atoms[r].begin(), atoms[r].end());
        out.shifts.insert(out.shifts.end(), images[r].begin(), images[r].end());
        for (const std::int64_t size : sizes[r]) {
            out.starts.push_back(out.starts.back() + size);
        }
    }
    return out;
}

}  // namespace mdapy::analysis

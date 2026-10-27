// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/dxa_mapping.h"

#include <cstddef>
#include <unordered_map>
#include <utility>

namespace mdapy::analysis::dxa {
namespace {

// the longest path of neighbors an ideal vector follows
constexpr int kPathSteps = 4;

// the corners of the edges, and the three edges around each face
constexpr int kEdges[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
constexpr int kFaces[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};

}  // namespace

Mapping::Mapping(const ClusterInput& atoms, const Clusters& clusters,
                 const Tetrahedra& tetrahedra, const Control& control)
    : atoms_(atoms), clusters_(clusters), tetrahedra_(tetrahedra), vertex_cluster_(clusters.cluster) {
    const std::int64_t n = atoms.n;
    const auto count = static_cast<std::int64_t>(tetrahedra.radii.size());
    // the atoms outside the clusters take the cluster of a neighbor along
    // the edges of the tetrahedra, as long as one is found
    for (bool changed = true; changed;) {
        changed = false;
        for (std::int64_t t = 0; t < count; ++t) {
            for (const auto& edge : kEdges) {
                const std::int64_t a = tetrahedra.atoms[static_cast<std::size_t>(4 * t + edge[0])];
                const std::int64_t b = tetrahedra.atoms[static_cast<std::size_t>(4 * t + edge[1])];
                std::int64_t& ca = vertex_cluster_[static_cast<std::size_t>(a)];
                std::int64_t& cb = vertex_cluster_[static_cast<std::size_t>(b)];
                if (ca < 0 && cb >= 0) {
                    ca = cb;
                    changed = true;
                } else if (cb < 0 && ca >= 0) {
                    cb = ca;
                    changed = true;
                }
            }
        }
    }
    // the crystal atoms each atom outside the clusters is a neighbor of
    reverse_offsets_.assign(static_cast<std::size_t>(n) + 1, 0);
    auto each_reverse = [&](auto&& visit) {
        for (std::int64_t a = 0; a < n; ++a) {
            if (clusters.cluster[static_cast<std::size_t>(a)] < 0) {
                continue;
            }
            const Lattice* l = lattice(atoms.structure[a]);
            for (int k = 0; k < l->neighbors; ++k) {
                const std::int64_t b = atoms.neighbors[16 * a + k];
                if (b >= 0 && clusters.cluster[static_cast<std::size_t>(b)] < 0) {
                    visit(a, k, b);
                }
            }
        }
    };
    each_reverse([&](std::int64_t, int, std::int64_t b) {
        ++reverse_offsets_[static_cast<std::size_t>(b) + 1];
    });
    for (std::int64_t i = 0; i < n; ++i) {
        reverse_offsets_[static_cast<std::size_t>(i) + 1] +=
            reverse_offsets_[static_cast<std::size_t>(i)];
    }
    reverse_atoms_.resize(static_cast<std::size_t>(reverse_offsets_.back()));
    reverse_places_.resize(reverse_atoms_.size());
    std::vector<std::int64_t> fill(reverse_offsets_.begin(), reverse_offsets_.end() - 1);
    each_reverse([&](std::int64_t a, int k, std::int64_t b) {
        const auto at = static_cast<std::size_t>(fill[static_cast<std::size_t>(b)]++);
        reverse_atoms_[at] = a;
        reverse_places_[at] = static_cast<std::int8_t>(k);
    });
    (void)control;
}

std::optional<Mat> Mapping::transition(std::int64_t c1, std::int64_t c2) const {
    if (c1 == c2) {
        return kIdentity;
    }
    for (const Transition& t : clusters_.transitions) {
        if (t.from == c1 && t.to == c2) {
            return t.matrix;
        }
        if (t.from == c2 && t.to == c1) {
            return transpose(t.matrix);
        }
    }
    if (clusters_.grain[static_cast<std::size_t>(c1)] ==
        clusters_.grain[static_cast<std::size_t>(c2)]) {
        Mat g1, g2;
        std::copy(clusters_.to_grain.begin() + 9 * c1, clusters_.to_grain.begin() + 9 * c1 + 9,
                  g1.begin());
        std::copy(clusters_.to_grain.begin() + 9 * c2, clusters_.to_grain.begin() + 9 * c2 + 9,
                  g2.begin());
        return multiply(transpose(g2), g1);
    }
    return std::nullopt;
}

namespace {

// the ideal vector of neighbor k of crystal atom a, in the frame of its
// cluster
Vec neighbor_vector(const ClusterInput& atoms, const Clusters& clusters, std::int64_t a, int k) {
    const Lattice* l = lattice(atoms.structure[a]);
    return turn(l->group[static_cast<std::size_t>(clusters.rotation[static_cast<std::size_t>(a)])],
                l->templates[static_cast<std::size_t>(atoms.templates[a])][static_cast<std::size_t>(k)]);
}

int place(const ClusterInput& atoms, std::int64_t a, std::int64_t b) {
    const Lattice* l = lattice(atoms.structure[a]);
    for (int k = 0; k < l->neighbors; ++k) {
        if (atoms.neighbors[16 * a + k] == b) {
            return k;
        }
    }
    return -1;
}

}  // namespace

std::optional<Vec> Mapping::ideal(std::int64_t i, std::int64_t j) const {
    const std::int64_t vi = vertex_cluster_[static_cast<std::size_t>(i)];
    const std::int64_t vj = vertex_cluster_[static_cast<std::size_t>(j)];
    if (vi < 0 || vj < 0) {
        return std::nullopt;
    }
    const std::int64_t ci = clusters_.cluster[static_cast<std::size_t>(i)];
    const std::int64_t cj = clusters_.cluster[static_cast<std::size_t>(j)];
    if (ci >= 0) {
        const int k = place(atoms_, i, j);
        if (k >= 0) {
            return neighbor_vector(atoms_, clusters_, i, k);
        }
    } else if (cj >= 0) {
        const int k = place(atoms_, j, i);
        if (k >= 0) {
            const auto m = transition(cj, vi);
            if (!m) {
                return std::nullopt;
            }
            const Vec v = turn(*m, neighbor_vector(atoms_, clusters_, j, k));
            return Vec{-v[0], -v[1], -v[2]};
        }
    }
    return path(i, j);
}

std::optional<Vec> Mapping::path(std::int64_t i, std::int64_t j) const {
    struct Node {
        std::int64_t atom;
        Vec vector;           // from i, in the frame of cluster frame
        std::int64_t frame;   // -1 before the first step
        int steps;
    };
    std::vector<Node> queue = {{i, {0, 0, 0}, -1, 0}};
    std::unordered_map<std::int64_t, bool> seen = {{i, true}};
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const Node current = queue[head];
        auto step = [&](std::int64_t next, const Vec& v, std::int64_t c) -> std::optional<Vec> {
            if (seen.count(next) != 0 || (current.steps >= kPathSteps - 1 && next != j)) {
                return std::nullopt;
            }
            Node node{next, v, c, current.steps + 1};
            if (current.frame >= 0) {
                const auto m = transition(c, current.frame);
                if (!m) {
                    return std::nullopt;
                }
                const Vec turned = turn(*m, v);
                node.vector = {current.vector[0] + turned[0], current.vector[1] + turned[1],
                               current.vector[2] + turned[2]};
                node.frame = current.frame;
            }
            if (next == j) {
                const std::int64_t vi = vertex_cluster_[static_cast<std::size_t>(i)];
                const auto m = transition(node.frame, vi);
                if (!m) {
                    return std::nullopt;
                }
                return turn(*m, node.vector);
            }
            seen[next] = true;
            queue.push_back(node);
            return std::nullopt;
        };
        const std::int64_t a = current.atom;
        const std::int64_t ca = clusters_.cluster[static_cast<std::size_t>(a)];
        if (ca >= 0) {
            const Lattice* l = lattice(atoms_.structure[a]);
            for (int k = 0; k < l->neighbors; ++k) {
                const std::int64_t b = atoms_.neighbors[16 * a + k];
                if (b < 0) {
                    continue;
                }
                if (const auto found = step(b, neighbor_vector(atoms_, clusters_, a, k), ca)) {
                    return found;
                }
            }
        } else {
            for (std::int64_t r = reverse_offsets_[static_cast<std::size_t>(a)];
                 r < reverse_offsets_[static_cast<std::size_t>(a) + 1]; ++r) {
                const std::int64_t b = reverse_atoms_[static_cast<std::size_t>(r)];
                const Vec v = neighbor_vector(atoms_, clusters_, b,
                                              reverse_places_[static_cast<std::size_t>(r)]);
                if (const auto found = step(b, {-v[0], -v[1], -v[2]},
                                            clusters_.cluster[static_cast<std::size_t>(b)])) {
                    return found;
                }
            }
        }
    }
    return std::nullopt;
}

bool Mapping::good(std::int64_t t) const {
    const std::int64_t* corner = tetrahedra_.atoms.data() + 4 * t;
    Vec edge[4][4];
    for (const auto& e : kEdges) {
        const auto v = ideal(corner[e[0]], corner[e[1]]);
        if (!v) {
            return false;
        }
        edge[e[0]][e[1]] = *v;
    }
    for (const auto& f : kFaces) {
        const std::int64_t c0 = vertex_cluster_[static_cast<std::size_t>(corner[f[0]])];
        const std::int64_t c1 = vertex_cluster_[static_cast<std::size_t>(corner[f[1]])];
        const std::int64_t c2 = vertex_cluster_[static_cast<std::size_t>(corner[f[2]])];
        const auto back = transition(c1, c0);
        if (!back) {
            return false;
        }
        // around the face, in the frame of its first corner
        const Vec second = turn(*back, edge[f[1]][f[2]]);
        const Vec sum = {edge[f[0]][f[1]][0] + second[0] - edge[f[0]][f[2]][0],
                         edge[f[0]][f[1]][1] + second[1] - edge[f[0]][f[2]][1],
                         edge[f[0]][f[1]][2] + second[2] - edge[f[0]][f[2]][2]};
        if (!close(sum, Vec{0, 0, 0})) {
            return false;
        }
        if (c0 != c1 || c1 != c2) {
            const auto t01 = transition(c0, c1), t12 = transition(c1, c2), t20 = transition(c2, c0);
            if (!t01 || !t12 || !t20 || !close(multiply(*t20, multiply(*t12, *t01)), kIdentity)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace mdapy::analysis::dxa

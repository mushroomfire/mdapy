// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Nearest neighbors with a k-d tree, for any triclinic box and any
// combination of periodic and non-periodic directions. The tree holds the
// atoms moved into the box and adapts to their density, so a cluster in
// vacuum or a slab costs no more than a crystal. Periodic images are searched
// by moving the query point by lattice vectors, as many as the neighbors
// need, so a box of any size works.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"
#include "neighbor/cell_list.h"
#include "neighbor/frame.h"

namespace mdapy::neighbor {

// A neighbor found by a search: atom j at the periodic image shift, so that
// the vector from the query point x is x_j - x + shift . H, of squared
// length r2.
struct Neighbor {
    double r2;
    index_t j;
    std::int32_t shift[3];
};

class KdTree {
public:
    // positions: n x 3; matrix: 3 x 3, rows a, b, c; origin: 3; pbc: 3.
    // Throws std::invalid_argument for a singular box, std::overflow_error
    // when n does not fit index_t.
    KdTree(const double* positions, std::int64_t n, const double* matrix, const double* origin,
           const bool* pbc);

    std::int64_t size() const { return static_cast<std::int64_t>(points_.size()); }

    // The k nearest neighbors of every atom of the tree, k per atom, ordered
    // by distance, then j, then shift; the pair (i, i, 0) is excluded.
    // Throws std::invalid_argument when k < 1 or when a system without
    // periodic directions has k atoms or fewer, std::overflow_error for a
    // shift outside int8.
    NeighborPairs nearest_atoms(int k, const Control& control = {}) const;

    // The k nearest atom images of each of m points (m x 3), ordered as
    // above: j (m x k), shifts (m x k x 3) and distances (m x k). Throws
    // std::invalid_argument when k < 1, when there are no atoms or when a
    // system without periodic directions has fewer than k atoms,
    // std::overflow_error for a shift outside int8.
    void nearest_points(const double* points, std::int64_t m, int k, index_t* j,
                        std::int8_t* shifts, double* distances, const Control& control = {}) const;

    // A point of the tree near a probe: its slot t, the periodic image of
    // its atom and the squared distance computed from the images in the box.
    struct Candidate {
        double r2;
        std::int64_t t;
        std::int32_t shift[3];
    };

private:
    struct Point {
        double y[3];  // the image of the atom in the box
        index_t atom;
    };
    struct Probe;
    class Search;

    void check_count(int k, std::int64_t available) const;
    // The length of the k-th shortest lattice vector.
    double lattice_reach(int k) const;
    // The k nearest neighbors of x, nearest first in found, the pair with
    // atom self and no shift excluded; reach bounds the distance of the
    // k-th neighbor beyond the farthest point of the tree. candidates is
    // working space.
    void search(const double* x, std::int64_t self, int k, double reach,
                std::vector<Candidate>& candidates, std::vector<Neighbor>& found) const;
    void scan(std::int64_t begin, std::int64_t end, const Probe& probe, Search& search) const;
    // Searches the node of count points from begin at depth, whose cell is
    // gap away from the probe along each axis, distance2 in all.
    void descend(std::int64_t begin, std::int64_t count, int depth, double* gap,
                 double distance2, const Probe& probe, Search& search) const;

    Frame frame_;
    bool periodic_;
    double size_ = 0.0;  // largest component of a position or its image in the box
    double low_[3], high_[3];  // the box of all points
    // per slot: the image of the atom in the box, its position and the
    // periodic images w between them (position - w . H is the image)
    std::vector<Point> points_;
    std::vector<double> positions_;
    std::vector<std::int32_t> wraps_;
};

// The k nearest neighbors of every atom, as KdTree::nearest_atoms.
NeighborPairs nearest(const double* positions, std::int64_t n, const double* matrix,
                      const double* origin, const bool* pbc, int k, const Control& control = {});

}  // namespace mdapy::neighbor

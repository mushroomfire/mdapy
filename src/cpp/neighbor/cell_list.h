// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Neighbor search with cells, for any triclinic box and any combination of
// periodic and non-periodic directions. Periodic images are enumerated, so a
// cutoff larger than the box is allowed: an atom then sees several images of
// the same neighbor, and images of itself.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::neighbor {

// The pairs (i, j, shift) with |x_j - x_i + shift . H| < cutoff, where H has
// the lattice vectors as rows and shift is an integer vector (zero along
// non-periodic directions); the pair (i, i, 0) is excluded. Pairs are
// grouped by i in CSR form, and sorted by j and then shift within a group,
// so the result does not depend on the number of threads.
struct NeighborPairs {
    std::vector<std::int64_t> offsets;    // n + 1: pairs of atom i are [offsets[i], offsets[i + 1])
    std::vector<std::int32_t> neighbors;  // j of each pair
    std::vector<std::int32_t> shifts;     // 3 per pair
    std::vector<double> distances;        // |r_ij| of each pair
};

// The atoms sorted into cells, and the search of the neighbors of an atom.
class CellGrid {
public:
    // positions: n x 3, kept by pointer; matrix: 3 x 3, rows a, b, c;
    // origin: 3; pbc: 3. Throws std::invalid_argument for a non-positive
    // cutoff or a singular box.
    CellGrid(const double* positions, std::int64_t n, const double* matrix, const double* origin,
             const bool* pbc, double cutoff);

    // Calls visit(j, shift, r2) for every pair (i, j, shift) closer than the
    // cutoff, shift an int32 array of 3; r2 is the squared distance.
    template <class Visit>
    void visit(std::int64_t i, Visit&& visit) const;

private:
    struct Axis {
        bool periodic = true;
        double low = 0.0;        // fractional coordinate where cell 0 starts
        double width = 1.0;      // fractional width of a cell
        std::int64_t cells = 1;
        std::int64_t reach = 1;  // cells to search on each side
    };

    static std::int64_t floor_div(std::int64_t a, std::int64_t b) {
        std::int64_t q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) {
            --q;
        }
        return q;
    }

    const double* positions_;
    double h_[9];
    double cutoff2_;
    Axis axes_[3];
    std::vector<std::int32_t> wraps_;        // 3 per atom
    std::vector<std::int64_t> cell_index_;   // 3 per atom
    std::vector<std::int64_t> cell_start_;   // per cell, and one past the end
    std::vector<std::int32_t> cell_atoms_;
};

template <class Visit>
void CellGrid::visit(std::int64_t i, Visit&& visit) const {
    const double* xi = positions_ + 3 * i;
    for (std::int64_t dx = -axes_[0].reach; dx <= axes_[0].reach; ++dx) {
        for (std::int64_t dy = -axes_[1].reach; dy <= axes_[1].reach; ++dy) {
            for (std::int64_t dz = -axes_[2].reach; dz <= axes_[2].reach; ++dz) {
                const std::int64_t step[3] = {dx, dy, dz};
                std::int64_t target = 0;
                std::int32_t image[3] = {0, 0, 0};
                bool inside = true;
                for (int d = 0; d < 3 && inside; ++d) {
                    const Axis& axis = axes_[d];
                    std::int64_t k = cell_index_[3 * i + d] + step[d];
                    if (axis.periodic) {
                        const std::int64_t wrap = floor_div(k, axis.cells);
                        k -= wrap * axis.cells;
                        image[d] = static_cast<std::int32_t>(wrap);
                    } else if (k < 0 || k >= axis.cells) {
                        inside = false;
                    }
                    target = target * axis.cells + k;
                }
                if (!inside) {
                    continue;
                }
                for (std::int64_t s = cell_start_[target]; s < cell_start_[target + 1]; ++s) {
                    const std::int32_t j = cell_atoms_[s];
                    // shift in the original (unwrapped) coordinates
                    const std::int32_t shift[3] = {image[0] + wraps_[3 * i] - wraps_[3 * j],
                                                   image[1] + wraps_[3 * i + 1] - wraps_[3 * j + 1],
                                                   image[2] + wraps_[3 * i + 2] - wraps_[3 * j + 2]};
                    if (j == i && shift[0] == 0 && shift[1] == 0 && shift[2] == 0) {
                        continue;
                    }
                    const double* xj = positions_ + 3 * j;
                    double r2 = 0.0;
                    for (int e = 0; e < 3; ++e) {
                        const double r = xj[e] - xi[e] + shift[0] * h_[e] + shift[1] * h_[3 + e] +
                                         shift[2] * h_[6 + e];
                        r2 += r * r;
                    }
                    if (r2 < cutoff2_) {
                        visit(j, shift, r2);
                    }
                }
            }
        }
    }
}

// positions: n x 3; matrix: 3 x 3, rows a, b, c; origin: 3; pbc: 3.
// Throws std::invalid_argument for a non-positive cutoff or a singular box.
NeighborPairs build(const double* positions, std::int64_t n, const double* matrix,
                    const double* origin, const bool* pbc, double cutoff,
                    const Control& control = {});

// The k nearest neighbors of every atom, nearest first (ties by neighbor and
// then shift), k per atom. A non-periodic system needs more than k atoms.
NeighborPairs nearest(const double* positions, std::int64_t n, const double* matrix,
                      const double* origin, const bool* pbc, int k, const Control& control = {});

// The vectors (pairs x 3) from each atom to its neighbors, x_j - x_i + shift . H,
// computed as the search computes the distances.
void pair_vectors(const double* positions, std::int64_t n, const std::int64_t* offsets,
                  const std::int32_t* neighbors, const std::int32_t* shifts, const double* matrix,
                  double* vectors, const Control& control = {});

// The pairs of each atom by distance, ties in their order, keeping the first
// count of each atom (all when count < 0): the positions of the kept pairs,
// in order, and the new offsets (n + 1).
void order_by_distance(std::int64_t n, const std::int64_t* offsets, const double* distances,
                       std::int64_t count, std::vector<std::int64_t>& order,
                       std::vector<std::int64_t>& kept, const Control& control = {});

}  // namespace mdapy::neighbor

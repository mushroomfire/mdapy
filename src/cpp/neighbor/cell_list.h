// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Neighbor search with cells, for any triclinic box and any combination of
// periodic and non-periodic directions. Periodic images are enumerated, so a
// cutoff larger than the box is allowed: an atom then sees several images of
// the same neighbor, and images of itself.

#pragma once

#include <cmath>
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
// The grid keeps its own copy of the atoms in cell order, so a search reads
// memory in sequence whatever the order of the atoms; searching the atoms in
// that order (atom(0), atom(1), ...) also reuses the cells just read.
class CellGrid {
public:
    // positions: n x 3; matrix: 3 x 3, rows a, b, c; origin: 3; pbc: 3.
    // Throws std::invalid_argument for a non-positive cutoff or a singular
    // box.
    CellGrid(const double* positions, std::int64_t n, const double* matrix, const double* origin,
             const bool* pbc, double cutoff);

    // The atom in slot t of the cell order.
    std::int32_t atom(std::int64_t t) const { return slot_atoms_[t]; }

    // Calls visit(j, shift, r2) for every pair (i, j, shift) closer than the
    // cutoff, shift an int32 array of 3; r2 is the squared distance.
    template <class Visit>
    void visit(std::int64_t i, Visit&& visit) const {
        const std::int64_t t = slot_of_[i];
        scan(&slot_cells_[3 * t], &slot_wraps_[3 * t], &slot_positions_[3 * t], i, visit);
    }

    // The same for any point x: every atom j and image shift with
    // |x_j + shift . H - x| below the cutoff.
    template <class Visit>
    void visit_point(const double* x, Visit&& visit) const;

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

    // atoms within reach of the cell cell (3) and the point x wrapped by wrap
    // (3); the pair with self and no shift is skipped
    template <class Visit>
    void scan(const std::int64_t* cell, const std::int32_t* wrap, const double* x,
              std::int64_t self, Visit&& visit) const;

    double h_[9];
    double inverse_[9];  // columns of H^-1
    double origin_[3];
    double cutoff2_;
    Axis axes_[3];
    std::vector<std::int64_t> cell_start_;  // first slot of each cell, and one past the end
    // per slot: the atom, its position, the periodic wraps that bring it
    // into the box and the three indices of its cell
    std::vector<std::int32_t> slot_atoms_;
    std::vector<double> slot_positions_;
    std::vector<std::int32_t> slot_wraps_;
    std::vector<std::int64_t> slot_cells_;
    std::vector<std::int32_t> slot_of_;  // per atom
};

template <class Visit>
void CellGrid::visit_point(const double* x, Visit&& visit) const {
    std::int64_t cell[3];
    std::int32_t wrap[3] = {0, 0, 0};
    const double r[3] = {x[0] - origin_[0], x[1] - origin_[1], x[2] - origin_[2]};
    for (int d = 0; d < 3; ++d) {
        const Axis& axis = axes_[d];
        double f = r[0] * inverse_[3 * d] + r[1] * inverse_[3 * d + 1] + r[2] * inverse_[3 * d + 2];
        if (axis.periodic) {
            const double k = std::floor(f);
            f -= k;
            wrap[d] = static_cast<std::int32_t>(k);
        }
        auto k = static_cast<std::int64_t>(std::floor((f - axis.low) / axis.width));
        cell[d] = k < 0 ? 0 : (k >= axis.cells ? axis.cells - 1 : k);
    }
    scan(cell, wrap, x, -1, visit);
}

template <class Visit>
void CellGrid::scan(const std::int64_t* cell, const std::int32_t* wrap, const double* xi,
                    std::int64_t self, Visit&& visit) const {
    for (std::int64_t dx = -axes_[0].reach; dx <= axes_[0].reach; ++dx) {
        for (std::int64_t dy = -axes_[1].reach; dy <= axes_[1].reach; ++dy) {
            for (std::int64_t dz = -axes_[2].reach; dz <= axes_[2].reach; ++dz) {
                const std::int64_t step[3] = {dx, dy, dz};
                std::int64_t target = 0;
                std::int32_t image[3] = {0, 0, 0};
                bool inside = true;
                for (int d = 0; d < 3 && inside; ++d) {
                    const Axis& axis = axes_[d];
                    std::int64_t k = cell[d] + step[d];
                    if (axis.periodic) {
                        const std::int64_t shift = floor_div(k, axis.cells);
                        k -= shift * axis.cells;
                        image[d] = static_cast<std::int32_t>(shift);
                    } else if (k < 0 || k >= axis.cells) {
                        inside = false;
                    }
                    target = target * axis.cells + k;
                }
                if (!inside) {
                    continue;
                }
                for (std::int64_t s = cell_start_[target]; s < cell_start_[target + 1]; ++s) {
                    // shift in the original (unwrapped) coordinates
                    const std::int32_t* wj = &slot_wraps_[3 * s];
                    const std::int32_t shift[3] = {image[0] + wrap[0] - wj[0],
                                                   image[1] + wrap[1] - wj[1],
                                                   image[2] + wrap[2] - wj[2]};
                    const double* xj = &slot_positions_[3 * s];
                    double r2 = 0.0;
                    for (int e = 0; e < 3; ++e) {
                        const double r = xj[e] - xi[e] + shift[0] * h_[e] + shift[1] * h_[3 + e] +
                                         shift[2] * h_[6 + e];
                        r2 += r * r;
                    }
                    if (r2 < cutoff2_) {
                        const std::int32_t j = slot_atoms_[s];
                        if (j == self && shift[0] == 0 && shift[1] == 0 && shift[2] == 0) {
                            continue;
                        }
                        visit(j, shift, r2);
                    }
                }
            }
        }
    }
}

// The nearest atom (site) to each of m points, with the squared distance;
// ties go to the lower index. sites: n x 3 in the box; points: m x 3.
void nearest_sites(const double* sites, std::int64_t n, const double* matrix,
                   const double* origin, const bool* pbc, const double* points, std::int64_t m,
                   std::int64_t* site, double* distance2, const Control& control = {});

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

// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Neighbor search with cells, for any triclinic box and any combination of
// periodic and non-periodic directions. Periodic images are enumerated, so a
// cutoff larger than the box is allowed: an atom then sees several images of
// the same neighbor, and images of itself.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/buffer.h"
#include "core/index.h"
#include "core/parallel.h"
#include "neighbor/frame.h"

namespace mdapy::neighbor {

// Pairs (i, j, shift) grouped by i in CSR form, shift an integer vector
// (zero along non-periodic directions) such that the vector of the pair is
// x_j - x_i + shift . H, H with the lattice vectors as rows.
struct NeighborPairs {
    Buffer<std::int64_t> offsets;  // n + 1: pairs of atom i are [offsets[i], offsets[i + 1])
    Buffer<index_t> neighbors;     // j of each pair
    Buffer<std::int8_t> shifts;    // 3 per pair
    Buffer<double> distances;      // |r_ij| of each pair
};

// The atoms sorted into cells, and the search of the neighbors of an atom.
// The grid keeps its own copy of the atoms in cell order (slots), so a search
// reads memory in sequence whatever the order of the atoms; searching the
// slots in order also reuses the cells just read.
class CellGrid {
public:
    // positions: n x 3; matrix: 3 x 3, rows a, b, c; origin: 3; pbc: 3.
    // Throws std::invalid_argument for a non-positive cutoff or a singular
    // box, std::overflow_error when n does not fit index_t.
    CellGrid(const double* positions, std::int64_t n, const double* matrix, const double* origin,
             const bool* pbc, double cutoff);

    // The atom in slot t.
    index_t atom(std::int64_t t) const { return slot_atoms_[t]; }

    // Calls visit(j, shift, r2) for every pair (i, j, shift) closer than the
    // cutoff, i the atom in slot t, shift an int32 array of 3 and r2 the
    // squared distance; the pair (i, i, 0) is skipped. The pairs come in an
    // order that depends on the positions, the box and the cutoff only.
    template <class Visit>
    void visit(std::int64_t t, Visit&& visit) const {
        scan<true>(t, visit);
    }

    // The number of pairs of the atom in slot t. Throws std::overflow_error
    // for a shift outside int8.
    std::int64_t count(std::int64_t t) const {
        std::int64_t pairs = 0;
        scan<false>(t, [&](index_t, const std::int32_t* shift, double) {
            check_shift(shift);
            ++pairs;
        });
        return pairs;
    }

private:
    struct Axis {
        bool periodic = true;
        double low = 0.0;        // fractional coordinate where cell 0 starts
        double width = 1.0;      // fractional width of a cell
        std::int64_t cells = 1;
        std::int64_t reach = 1;  // cells to search on each side
    };

    // visit, but without kExact the pairs far enough inside the cutoff
    // come with the squared distance of the screen, not the exact one.
    template <bool kExact, class Visit>
    void scan(std::int64_t t, Visit&& visit) const;

    static std::int64_t floor_div(std::int64_t a, std::int64_t b) {
        std::int64_t q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) {
            --q;
        }
        return q;
    }

    Frame frame_;
    double cutoff2_;
    // the square of the cutoff enlarged, and reduced, by a margin far above
    // the rounding of a distance between positions in the box and between
    // the originals
    double screen2_, inner2_;
    Axis axes_[3];
    std::vector<std::int64_t> cell_start_;  // first slot of each cell, and one past the end
    // per slot: the atom, its position, its image in the box, the periodic
    // images w between them (position - w . H is in the box) and its cell
    std::vector<index_t> slot_atoms_;
    std::vector<double> slot_positions_;
    std::vector<double> slot_wrapped_;
    std::vector<std::int32_t> slot_wraps_;
    std::vector<std::int64_t> slot_cells_;
};

template <bool kExact, class Visit>
void CellGrid::scan(std::int64_t t, Visit&& visit) const {
    const double* h = frame_.h;
    const double* xi = &slot_positions_[3 * t];
    const std::int32_t* wrap = &slot_wraps_[3 * t];
    const index_t self = slot_atoms_[t];
    std::int64_t cell[3];
    for (std::int64_t c = slot_cells_[t], d = 2; d >= 0; --d) {
        cell[d] = c % axes_[d].cells;
        c /= axes_[d].cells;
    }
    // Candidates are screened with the positions in the box, whose image
    // offset is the same for a whole cell; the pairs that pass are measured
    // exactly as documented. The cells along the last direction are read in
    // runs, each run a range of slots in one periodic image.
    const double* xw = &slot_wrapped_[3 * t];
    const Axis& last = axes_[2];
    for (std::int64_t dx = -axes_[0].reach; dx <= axes_[0].reach; ++dx) {
        for (std::int64_t dy = -axes_[1].reach; dy <= axes_[1].reach; ++dy) {
            const std::int64_t step[2] = {dx, dy};
            std::int64_t row = 0;
            std::int32_t image[3] = {0, 0, 0};
            bool inside = true;
            for (int d = 0; d < 2 && inside; ++d) {
                const Axis& axis = axes_[d];
                std::int64_t k = cell[d] + step[d];
                if (axis.periodic) {
                    const std::int64_t shift = floor_div(k, axis.cells);
                    k -= shift * axis.cells;
                    image[d] = static_cast<std::int32_t>(shift);
                } else if (k < 0 || k >= axis.cells) {
                    inside = false;
                }
                row = row * axis.cells + k;
            }
            if (!inside) {
                continue;
            }
            std::int64_t z = cell[2] - last.reach, z_end = cell[2] + last.reach;
            if (!last.periodic) {
                z = std::max<std::int64_t>(z, 0);
                z_end = std::min(z_end, last.cells - 1);
            }
            while (z <= z_end) {
                const std::int64_t shift = last.periodic ? floor_div(z, last.cells) : 0;
                const std::int64_t k = z - shift * last.cells;
                const std::int64_t run = std::min(z_end - z, last.cells - 1 - k);
                const std::int64_t first = row * last.cells + k;
                z += run + 1;
                image[2] = static_cast<std::int32_t>(shift);
                double base[3];
                for (int e = 0; e < 3; ++e) {
                    base[e] = xw[e] - (image[0] * h[e] + image[1] * h[3 + e] + image[2] * h[6 + e]);
                }
                for (std::int64_t s = cell_start_[first]; s < cell_start_[first + run + 1]; ++s) {
                    const double* yj = &slot_wrapped_[3 * s];
                    const double ax = yj[0] - base[0], ay = yj[1] - base[1], az = yj[2] - base[2];
                    const double a2 = ax * ax + ay * ay + az * az;
                    if (a2 >= screen2_) {
                        continue;
                    }
                    // shift in the original coordinates
                    const std::int32_t* wj = &slot_wraps_[3 * s];
                    const std::int32_t pair_shift[3] = {image[0] + wrap[0] - wj[0],
                                                        image[1] + wrap[1] - wj[1],
                                                        image[2] + wrap[2] - wj[2]};
                    if (!kExact && a2 < inner2_) {
                        if (s != t || pair_shift[0] != 0 || pair_shift[1] != 0 ||
                            pair_shift[2] != 0) {
                            visit(slot_atoms_[s], pair_shift, a2);
                        }
                        continue;
                    }
                    double r[3];
                    const double r2 = pair_vector(xi, &slot_positions_[3 * s], pair_shift, h, r);
                    if (r2 < cutoff2_) {
                        const index_t j = slot_atoms_[s];
                        if (j == self && pair_shift[0] == 0 && pair_shift[1] == 0 &&
                            pair_shift[2] == 0) {
                            continue;
                        }
                        visit(j, pair_shift, r2);
                    }
                }
            }
        }
    }
}

// Every pair closer than the cutoff, the pair (i, i, 0) excluded; within an
// atom the pairs come in the order of CellGrid::visit.
// positions: n x 3; matrix: 3 x 3, rows a, b, c; origin: 3; pbc: 3.
// Throws std::invalid_argument for a non-positive cutoff or a singular box,
// std::overflow_error for a shift outside int8 or too many atoms.
NeighborPairs build(const double* positions, std::int64_t n, const double* matrix,
                    const double* origin, const bool* pbc, double cutoff,
                    const Control& control = {});

// The vectors (pairs x 3) from each atom to its neighbors, x_j - x_i + shift . H,
// computed as the search computes the distances.
void pair_vectors(const double* positions, std::int64_t n, const std::int64_t* offsets,
                  const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                  double* vectors, const Control& control = {});

// The pairs of a list (offsets, neighbors, shifts) that are closer than the
// cutoff with the atoms at positions, in the order of the list, with their
// distances and vectors (pairs x 3); the test and the distances are those of
// the search, so a list from a search with a longer cutoff gives the pairs
// of a new search when no atom has moved half the difference of the cutoffs.
NeighborPairs follow(const double* positions, std::int64_t n, const std::int64_t* offsets,
                     const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                     double cutoff, Buffer<double>& vectors, const Control& control = {});

// The pairs of each of n atoms (positions: n x 3; matrix: rows a, b, c) by
// their squared distance from pair_vector, ties by atom and then image, as
// a search of nearest neighbors orders them, keeping the first count of each
// atom (all when count < 0): the positions of the kept pairs, in order, and
// the new offsets (n + 1).
void order_by_distance(const double* positions, std::int64_t n, const std::int64_t* offsets,
                       const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                       std::int64_t count, std::vector<std::int64_t>& order,
                       std::vector<std::int64_t>& kept, const Control& control = {});

}  // namespace mdapy::neighbor

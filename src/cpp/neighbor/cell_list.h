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
    index_t atom(std::int64_t t) const { return slots_[t].atom; }

    // Calls visit(j, shift, r2) for every pair (i, j, shift) closer than the
    // cutoff, i the atom in slot t, shift an int32 array of 3 and r2 the
    // squared distance; the pair (i, i, 0) is skipped. The pairs come in an
    // order that depends on the positions, the box and the cutoff only.
    template <class Visit>
    void visit(std::int64_t t, Visit&& visit) const {
        scan<true, false>(t, visit);
    }

    // Calls visit(s, shift) for the same pairs as visit, in the same order,
    // with the slot s of the second atom instead of the atom and without
    // their distances: data of the atoms copied into slot order is read
    // nearly in sequence.
    template <class Visit>
    void visit_slots(std::int64_t t, Visit&& visit) const {
        scan<false, true>(t, visit);
    }

    // Calls visit(s, shift, r2) for the pairs of visit, in the same order,
    // with the slot s of the second atom and the squared distance of visit.
    template <class Visit>
    void visit_slots_measured(std::int64_t t, Visit&& visit) const {
        scan<true, true>(t, visit);
    }

    // The number of pairs of the atom in slot t. Throws std::overflow_error
    // for a shift outside int8.
    std::int64_t count(std::int64_t t) const {
        std::int64_t pairs = 0;
        scan<false, false>(t, [&](index_t, const std::int32_t* shift, double) {
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
    // come with the squared distance of the screen, not the exact one; with
    // kSlots, visit(s, shift) gets the slot of the second atom, and with both
    // visit(s, shift, r2).
    template <bool kExact, bool kSlots, class Visit>
    void scan(std::int64_t t, Visit&& visit) const;

    // floor(k / cells): the periodic image of cell k, without a division when
    // it is next to the box, as it nearly always is
    static std::int64_t wrap_shift(std::int64_t k, std::int64_t cells) {
        if (k >= 0 && k < cells) {
            return 0;
        }
        if (k < 0 && k >= -cells) {
            return -1;
        }
        if (k >= cells && k < 2 * cells) {
            return 1;
        }
        return floor_div(k, cells);
    }

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
    Buffer<std::int64_t> cell_start_;  // first slot of each cell, and one past the end
    // per slot: the atom, its position, its image in the box, the periodic
    // images w between them (position - w . H is in the box) and its cell
    // what a search reads of a slot, in one cache line
    struct alignas(64) Slot {
        double wrapped[3];      // the image of the atom in the box
        double position[3];
        std::int32_t wraps[3];  // periodic images w with position - w . H in the box
        index_t atom;
    };
    Buffer<Slot> slots_;
    Buffer<std::int32_t> slot_cells_;  // the cell of each slot, 3 coordinates
};

template <bool kExact, bool kSlots, class Visit>
void CellGrid::scan(std::int64_t t, Visit&& visit) const {
    const double* h = frame_.h;
    const Slot& own = slots_[t];
    const double* xi = own.position;
    const std::int32_t* wrap = own.wraps;
    const index_t self = own.atom;
    const std::int64_t cell[3] = {slot_cells_[3 * t], slot_cells_[3 * t + 1],
                                  slot_cells_[3 * t + 2]};
    // Candidates are screened with the positions in the box, whose image
    // offset is the same for a whole cell; the pairs that pass are measured
    // exactly as documented. The cells along the first direction, which
    // varies fastest in the order of the cells, are read in runs, each run a
    // range of slots in one periodic image.
    const double* xw = own.wrapped;
    const Axis& last = axes_[0];
    for (std::int64_t dz = -axes_[2].reach; dz <= axes_[2].reach; ++dz) {
        for (std::int64_t dy = -axes_[1].reach; dy <= axes_[1].reach; ++dy) {
            const std::int64_t step[3] = {0, dy, dz};
            std::int64_t row = 0;
            std::int32_t image[3] = {0, 0, 0};
            bool inside = true;
            for (int d = 2; d >= 1 && inside; --d) {
                const Axis& axis = axes_[d];
                std::int64_t k = cell[d] + step[d];
                if (axis.periodic) {
                    const std::int64_t shift = wrap_shift(k, axis.cells);
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
            std::int64_t z = cell[0] - last.reach, z_end = cell[0] + last.reach;
            if (!last.periodic) {
                z = std::max<std::int64_t>(z, 0);
                z_end = std::min(z_end, last.cells - 1);
            }
            while (z <= z_end) {
                const std::int64_t shift = last.periodic ? wrap_shift(z, last.cells) : 0;
                const std::int64_t k = z - shift * last.cells;
                const std::int64_t run = std::min(z_end - z, last.cells - 1 - k);
                const std::int64_t first = row * last.cells + k;
                z += run + 1;
                image[0] = static_cast<std::int32_t>(shift);
                double base[3] = {xw[0], xw[1], xw[2]};
                if ((image[0] | image[1] | image[2]) != 0) {
                    for (int e = 0; e < 3; ++e) {
                        base[e] -= image[0] * h[e] + image[1] * h[3 + e] + image[2] * h[6 + e];
                    }
                }
                for (std::int64_t s = cell_start_[first]; s < cell_start_[first + run + 1]; ++s) {
                    const Slot& other = slots_[s];
                    const double* yj = other.wrapped;
                    const double ax = yj[0] - base[0], ay = yj[1] - base[1], az = yj[2] - base[2];
                    const double a2 = ax * ax + ay * ay + az * az;
                    if (a2 >= screen2_) {
                        continue;
                    }
                    // shift in the original coordinates
                    const std::int32_t* wj = other.wraps;
                    const std::int32_t pair_shift[3] = {image[0] + wrap[0] - wj[0],
                                                        image[1] + wrap[1] - wj[1],
                                                        image[2] + wrap[2] - wj[2]};
                    if constexpr (!kExact) {
                        if (a2 < inner2_) {
                            if (s != t || pair_shift[0] != 0 || pair_shift[1] != 0 ||
                                pair_shift[2] != 0) {
                                if constexpr (kSlots) {
                                    visit(s, pair_shift);
                                } else {
                                    visit(other.atom, pair_shift, a2);
                                }
                            }
                            continue;
                        }
                    }
                    double r[3];
                    const double r2 = pair_vector(xi, other.position, pair_shift, h, r);
                    if (r2 < cutoff2_) {
                        const index_t j = other.atom;
                        if (j == self && pair_shift[0] == 0 && pair_shift[1] == 0 &&
                            pair_shift[2] == 0) {
                            continue;
                        }
                        if constexpr (kSlots && kExact) {
                            visit(s, pair_shift, r2);
                        } else if constexpr (kSlots) {
                            visit(s, pair_shift);
                        } else {
                            visit(j, pair_shift, r2);
                        }
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

// The pairs of a list closer than their limit, in the order of the list:
// pair (i, j) is kept when its distance is below limits[type[i] * types +
// type[j]] (limits: types x types), or below limits[0] when type is null.
NeighborPairs within(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                     const std::int8_t* shifts, const double* distances,
                     const std::int32_t* type, int types, const double* limits,
                     const Control& control = {});

// The pairs of a list (offsets, neighbors, shifts) that are closer than the
// cutoff with the atoms at positions, in the order of the list, with their
// distances; the test and the distances are those of the search, so a list
// from a search with a longer cutoff gives the pairs of a new search when no
// atom has moved half the difference of the cutoffs.
NeighborPairs follow(const double* positions, std::int64_t n, const std::int64_t* offsets,
                     const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                     double cutoff, const Control& control = {});

// The pairs of each of n atoms (positions: n x 3; matrix: rows a, b, c) by
// their squared distance from pair_vector, ties by atom and then image, as
// a search of nearest neighbors orders them, keeping the first count of each
// atom (all when count < 0): the positions of the kept pairs, in order, and
// the new offsets (n + 1).
void order_by_distance(const double* positions, std::int64_t n, const std::int64_t* offsets,
                       const index_t* neighbors, const std::int8_t* shifts, const double* matrix,
                       std::int64_t count, std::vector<std::int64_t>& order,
                       std::vector<std::int64_t>& kept, const Control& control = {});

// The same order in place: the first count pairs of each of n atoms (all
// when count < 0) come first, in the order of a search of nearest
// neighbors, the pairs of each atom permuted among themselves with their
// shifts and distances.
void sort_nearest(const double* positions, std::int64_t n, const std::int64_t* offsets,
                  index_t* neighbors, std::int8_t* shifts, double* distances,
                  const double* matrix, std::int64_t count, const Control& control = {});

}  // namespace mdapy::neighbor

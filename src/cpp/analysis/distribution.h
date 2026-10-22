// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Histograms of pair distances, counted during the neighbor search without
// storing the pairs.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::analysis {

// The neighbors of each of n atoms counted by type (counts: n x types), those
// closer than the limit of their pair of types (limits: types x types, in Å;
// none for 0), found on a cell grid. With per_pair false every limit is the
// same and the squared distance is compared with its square, as a neighbor
// search does; with per_pair true the distance is compared with each limit,
// as NeighborList.within does.
void count_neighbors(const double* positions, std::int64_t n, const double* matrix,
                     const double* origin, const bool* pbc, const std::int32_t* type, int types,
                     const double* limits, bool per_pair, std::int64_t* counts,
                     const Control& control = {});

// Counts of the ordered pairs (i, j) closer than r_max in bins of equal width,
// by the types of i and j: counts[(type_i * types + type_j) * bins + bin].
// positions: n x 3; matrix: rows a, b, c; types: n values from 0 to types - 1.
std::vector<std::int64_t> pair_histogram(const double* positions, std::int64_t n,
                                         const double* matrix, const double* origin,
                                         const bool* pbc, double r_max, int bins,
                                         const std::int32_t* type, int types,
                                         const Control& control = {});

// Static structure factor of a periodic box at every reciprocal lattice
// vector q = h b1 + k b2 + l b3 with 0 < |q| <= q_max, one of each pair +q,
// -q: sums over bins of equal width of Re(rho_a(q) rho_b(q)*) for the types a,
// b (types x types x bins, rho_t the sum of exp(i q . r) over the atoms of
// type t), and the number of vectors in each bin. matrix: rows a1, a2, a3.
// The atoms are taken in blocks of block atoms (0: as many as fit 256 MB of
// tables).
void structure_factor(const double* positions, std::int64_t n, const double* matrix,
                      const std::int32_t* type, int types, double q_max, int bins,
                      std::vector<double>& sums, std::vector<std::int64_t>& vectors,
                      std::int64_t block = 0, const Control& control = {});

}  // namespace mdapy::analysis

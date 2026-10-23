// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Identification of local crystal structure: the centro-symmetry parameter
// and common neighbor analysis.

#pragma once

#include <cstdint>

#include "core/index.h"
#include "core/parallel.h"
#include "neighbor/list_view.h"

namespace mdapy::identify {

// Structure types of common neighbor analysis.
enum Structure : std::int8_t { kOther = 0, kFcc = 1, kHcp = 2, kBcc = 3, kIco = 4 };

// The centro-symmetry parameter of n atoms from the first k pairs of each
// in list (k even), its nearest: the sum of the k / 2 smallest
// |r_a + r_b|^2 over the pairs of neighbors a < b; NaN for an atom with
// fewer pairs.
void centro_symmetry(const neighbor::ListView& list, std::int64_t n, int k, double* csp,
                     const Control& control = {});

// The common neighbor parameter (Tsuzuki, Branicio and Rino) of n atoms
// from their pairs in list closer than cutoff: for each neighbor j the sum of
// r_ik + r_jk over the neighbors k of i that are closer than the cutoff to
// j, squared and averaged over j, in Å². NaN for an atom without neighbors.
void common_neighbor_parameter(const neighbor::ListView& list, std::int64_t n, double cutoff,
                               double* cnp, const Control& control = {});

// The same with the neighbors found on a cell grid (positions: n x 3;
// matrix: rows a, b, c; origin: 3; pbc: 3).
void common_neighbor_parameter_search(const double* positions, std::int64_t n,
                                      const double* matrix, const double* origin,
                                      const bool* pbc, double cutoff, double* cnp,
                                      const Control& control = {});

// Common neighbor analysis of n atoms from their pairs in list. With a
// cutoff of the list the neighbors are those within it; otherwise the cutoff
// of each atom is adaptive and the pairs of the list must be its nearest,
// nearest first, at least 14 of them for bcc to be found.
void common_neighbor_analysis(const neighbor::ListView& list, std::int64_t n,
                              std::int8_t* types, const Control& control = {});

// The same with a cutoff, with the neighbors found on a cell grid
// (positions: n x 3; matrix: rows a, b, c; origin: 3; pbc: 3).
void common_neighbor_analysis_search(const double* positions, std::int64_t n,
                                     const double* matrix, const double* origin, const bool* pbc,
                                     double cutoff, std::int8_t* types,
                                     const Control& control = {});

// Ackland-Jones analysis (Ackland and Jones 2006, the rules of its original
// implementation) of n atoms from their pairs in list, their nearest
// neighbors, nearest first, of which the first 14 count; fewer than 6 give
// other.
void ackland_jones(const neighbor::ListView& list, std::int64_t n, std::int8_t* types,
                   const Control& control = {});

// centro_symmetry (k from 2 to 14), common_neighbor_analysis with adaptive
// cutoffs and ackland_jones (k from 1 to 14), each with the k nearest
// neighbors of every atom found by a k-d tree as it goes, without a list
// (positions: n x 3; matrix: rows a, b, c; origin: 3; pbc: 3). They throw as
// a search of the k nearest neighbors does.
void centro_symmetry_nearest(const double* positions, std::int64_t n, const double* matrix,
                             const double* origin, const bool* pbc, int k, double* csp,
                             const Control& control = {});
void common_neighbor_analysis_nearest(const double* positions, std::int64_t n,
                                      const double* matrix, const double* origin,
                                      const bool* pbc, int k, std::int8_t* types,
                                      const Control& control = {});
void ackland_jones_nearest(const double* positions, std::int64_t n, const double* matrix,
                           const double* origin, const bool* pbc, int k, std::int8_t* types,
                           const Control& control = {});

// Diamond structures: 0 other, 1 cubic, 2 hexagonal.
enum Diamond : std::int8_t { kNotDiamond = 0, kCubicDiamond = 1, kHexagonalDiamond = 2 };

// Cubic or hexagonal diamond for each of n atoms from its four nearest
// neighbors, the first four pairs of each atom in list, nearest first: the
// twelve second neighbors, reached through the first ones, must form fcc
// (cubic) or hcp (hexagonal) under adaptive common neighbor analysis.
void identify_diamond(const neighbor::ListView& list, std::int64_t n, std::int8_t* types,
                      const Control& control = {});

// CHILL+ types of water molecules: 0 other, 1 hexagonal ice, 2 cubic ice,
// 3 interfacial ice, 4 hydrate, 5 interfacial hydrate.
enum Water : std::int8_t {
    kLiquid = 0,
    kHexagonalIce = 1,
    kCubicIce = 2,
    kInterfacialIce = 3,
    kHydrate = 4,
    kInterfacialHydrate = 5
};

// CHILL+ type of each of n molecules from their pairs in list: bonds are
// staggered or eclipsed by the correlation of the q_3m of their two
// molecules.
void chill_plus(const neighbor::ListView& list, std::int64_t n, std::int8_t* types,
                const Control& control = {});

}  // namespace mdapy::identify

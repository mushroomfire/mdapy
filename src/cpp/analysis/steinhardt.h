// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Steinhardt bond-orientational order parameters.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"
#include "neighbor/list_view.h"

namespace mdapy::analysis {

constexpr int kMaxDegree = 20;

// Real orthonormal spherical harmonics of the unit vector u for degree l up
// to kMaxDegree, m = 0, then the cosine and sine parts of m = 1..l: 2 l + 1
// values.
void real_harmonics(int l, const double* u, double* y);

// The order parameters q_l of n atoms for each degree in degrees, from
// their pairs in list, each weighted by weights at its pair (P values), or
// all alike when weights is null. With average, the harmonics of each atom
// are averaged with those of its neighbors in list first. q is (n, degrees).
void steinhardt(const neighbor::ListView& list, std::int64_t n, const double* weights,
                const std::vector<int>& degrees, bool average, double* q,
                const Control& control = {});

// The number of solid-like bonds of each atom (ten Wolde, Ruiz-Montero and
// Frenkel): a pair in list is solid-like when the product of the harmonics
// of degree l of its two atoms, each normalized to length 1, exceeds
// threshold. The harmonics of an atom are those of its pairs in list, each
// weighted by weights at its pair, or all alike when weights is null; an
// atom without pairs has none. bonds: n.
void solid_bonds(const neighbor::ListView& list, std::int64_t n, const double* weights, int l,
                 double threshold, std::int32_t* bonds, const Control& control = {});

// steinhardt without averaging, with the k nearest neighbors of every atom
// found by a k-d tree as it goes, without a list (positions: n x 3; matrix:
// rows a, b, c; origin: 3; pbc: 3). Throws as a search of the k nearest
// neighbors does.
void steinhardt_nearest(const double* positions, std::int64_t n, const double* matrix,
                        const double* origin, const bool* pbc, int k,
                        const std::vector<int>& degrees, double* q, const Control& control = {});

}  // namespace mdapy::analysis

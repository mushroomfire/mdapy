// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The Tersoff potential, as LAMMPS pair_style tersoff computes it (Tersoff,
// Phys. Rev. B 37, 6991 (1988) and 39, 5566 (1989)).

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

// The parameters of a triplet of elements i, j, k, as a line of a .tersoff
// file: m gamma lambda3 c d cos(theta0) n beta lambda2 B R D lambda1 A.
struct TersoffParameters {
    int m = 3;
    double gamma = 0, lam3 = 0, c = 0, d = 0, h = 0, n = 0, beta = 0, lam2 = 0, b = 0, r = 0,
           big_d = 0, lam1 = 0, a = 0;
    double cut = 0, cutsq = 0;  // R + D
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;  // where the bond order is approximated
};

struct TersoffPotential {
    int elements = 0;
    double cutoff = 0.0;  // the largest cutoff
    // elements^3 triplets, (i * elements + j) * elements + k
    std::vector<TersoffParameters> parameters;

    const TersoffParameters& of(int i, int j, int k) const {
        return parameters[static_cast<std::size_t>((i * elements + j) * elements + k)];
    }
};

// Builds a potential from rows of 14 numbers, elements^3 of them in the
// order of TersoffPotential::parameters.
TersoffPotential make_tersoff(int elements, const double* rows);

// Energies (n), forces (n x 3) and virials (n x 9) of n atoms, with a full
// neighbor list in CSR form whose pair vectors are those of
// neighbor::pair_vector, the energies and virials shared among the atoms
// as LAMMPS tallies them; heat: those of the heat flux instead, each term
// whole with its centre and the virials of Fan et al., PRB 92, 094301
// (2015), which differ from LAMMPS's atom by atom but not in their sums.
void compute_tersoff(const TersoffPotential& potential, std::int64_t n,
                     const std::int32_t* species, const double* positions, const double* matrix,
                     const std::int64_t* offsets, const index_t* neighbors,
                     const std::int8_t* shifts, double* energies, double* forces,
                     double* virials, const Control& control = {}, bool heat = false,
                     std::int64_t centers = -1,
                     const std::int64_t* reverse = nullptr);

}  // namespace mdapy::calculators

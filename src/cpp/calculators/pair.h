// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Pair potentials of one formula and coefficients for each pair of species:
// Lennard-Jones, Morse and Uhlenbeck-Ford, as LAMMPS pair_style lj/cut, morse and
// ufm compute them.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

enum class PairStyle { LennardJones, Morse, UhlenbeckFord };

struct PairPotential {
    PairStyle style = PairStyle::LennardJones;
    int species = 0;
    double cutoff = 0.0;          // the largest cutoff, in Å
    std::vector<double> cutsq;    // species x species, squared cutoffs
    std::vector<double> offset;   // species x species, energies taken off
    // species x species x 4: Lennard-Jones 48 e s^12, 24 e s^6, 4 e s^12,
    // 4 e s^6; Morse D0, alpha, r0, 2 D0 alpha; Uhlenbeck-Ford 2 e / s^2,
    // 1 / s^2, e, 0
    std::vector<double> coefficients;
};

// Energies (n), forces (n x 3) and virials (n x 9) of n atoms, with a full
// neighbor list in CSR form whose pair vectors are those of
// neighbor::pair_vector. Each atom gathers its own pairs; the energy and the
// virial of each pair are shared equally by its two atoms.
// The potential of tables of species x species entries: squared cutoffs,
// energies taken off and 4 coefficients each. Throws std::invalid_argument
// for negative cutoffs.
PairPotential make_pair_potential(PairStyle style, int species, const double* cutsq,
                                  const double* offset, const double* coefficients);

void compute_pair(const PairPotential& potential, std::int64_t n, const std::int32_t* species,
                  const double* positions, const double* matrix, const std::int64_t* offsets,
                  const index_t* neighbors, const std::int8_t* shifts, double* energies,
                  double* forces, double* virials, const Control& control = {});

}  // namespace mdapy::calculators

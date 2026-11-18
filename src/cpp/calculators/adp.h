// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The angular-dependent potential (ADP) of tabulated functions, as LAMMPS
// pair_style adp computes it (Mishin, Mehl and Papaconstantopoulos, Acta
// Mater. 53, 4029 (2005)): the embedded atom method with the dipole and
// quadrupole of the neighbors of each atom.

#pragma once

#include <cstdint>
#include <vector>

#include "calculators/eam.h"
#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

struct AdpPotential {
    int elements = 0;
    double cutoff = 0.0;
    std::vector<Spline> embedding;   // F(rho), one per element
    std::vector<Spline> density;     // rho(r) an atom of each element gives
    // one per pair of elements a >= b, a (a + 1) / 2 + b: r phi(r), u(r), w(r)
    std::vector<Spline> pair, dipole, quadrupole;

    std::size_t pair_index(int a, int b) const {
        if (a < b) {
            const int t = a;
            a = b;
            b = t;
        }
        return static_cast<std::size_t>(a * (a + 1) / 2 + b);
    }
};

// Energies (n), forces (n x 3) and virials (n x 9) of n atoms, with a full
// neighbor list in CSR form whose pair vectors are those of
// neighbor::pair_vector; the energy and the virial of each pair shared
// equally by its two atoms, as LAMMPS.
void compute_adp(const AdpPotential& potential, std::int64_t n, const std::int32_t* species,
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, const Control& control = {});

}  // namespace mdapy::calculators

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

#include <cstddef>
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
// The potential of tables: embedding and density (elements, nrho or nr);
// pair, dipole and quadrupole (elements * (elements + 1) / 2, nr), the lower
// triangle by rows. Throws std::invalid_argument for tables that do not fit.
AdpPotential make_adp_potential(const double* embedding, int elements, int nrho,
                                const double* density, const double* pair, const double* dipole,
                                const double* quadrupole, int nr, double drho, double dr,
                                double cutoff);

void compute_adp(const AdpPotential& potential, std::int64_t n, const std::int32_t* species,
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, const Control& control = {});

// What each atom has of its neighbors in ADP, F'(rho), its dipole (3) and
// its quadrupole (xx yy zz yz xz xy): kState numbers in a row.
inline constexpr std::size_t kState = 10;

// The two passes of compute_adp for the first centers atoms, as the processes
// of MPI split them: their embedding and angular energies and their states;
// then, given the states of every atom of the neighbor list, the forces on
// the centres, their pair energies added, and their virials unless null.
void adp_densities(const AdpPotential& potential, std::int64_t centers,
                   const std::int32_t* species, const double* positions, const double* matrix,
                   const std::int64_t* offsets, const index_t* neighbors,
                   const std::int8_t* shifts, double* energies, double* state,
                   const Control& control = {});
void adp_forces(const AdpPotential& potential, std::int64_t centers, const std::int32_t* species,
                const double* positions, const double* matrix, const std::int64_t* offsets,
                const index_t* neighbors, const std::int8_t* shifts, const double* state,
                double* energies, double* forces, double* virials, const Control& control = {});

}  // namespace mdapy::calculators

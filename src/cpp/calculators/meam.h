// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The modified embedded atom method (MEAM), as LAMMPS pair_style meam
// computes it (Baskes, Phys. Rev. B 46, 2727 (1992); Lee and Baskes, Phys.
// Rev. B 62, 8564 (2000)): the parameters and the tables of the pair
// functions come set up from Python; here the screening, the partial
// densities, the energies, the forces and the virials.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

struct MeamPotential {
    int elements = 0;
    int ialloy = 0, mix_ref_t = 0, emb_lin_neg = 0, bkgd_dyn = 0;
    double rc = 4.0, delr = 0.1, gsmooth_factor = 99.0, dr = 0.0;
    double cutoff = 0.0;  // the radius of the neighbors needed
    int nr = 0;
    // per element, rows: rho0, beta0..3, t1..t3, A, Ec, re, rho_ref, Z, ibar,
    // the three shape factors
    std::vector<double> element;
    std::vector<double> ebound;      // elements^2
    std::vector<double> cmin, cmax;  // elements^3, [i][j][k], k the screening atom
    // per pair of elements a <= b, 7 rows of nr: phi and its spline
    std::vector<double> table;

    enum Row { RHO0, BETA0, BETA1, BETA2, BETA3, T1, T2, T3, A, EC, RE, RHO_REF, Z, IBAR, S1, S2, S3 };

    double of(Row row, int a) const {
        return element[static_cast<std::size_t>(row) * static_cast<std::size_t>(elements) +
                       static_cast<std::size_t>(a)];
    }
    std::size_t triple(int i, int j, int k) const {
        return static_cast<std::size_t>((i * elements + j) * elements + k);
    }
    const double* pair_table(int a, int b) const {
        if (a > b) {
            const int t = a;
            a = b;
            b = t;
        }
        const int index = a * elements - a * (a - 1) / 2 + (b - a);
        return table.data() + static_cast<std::size_t>(index) * 7 * static_cast<std::size_t>(nr);
    }
};

// Energies (n), forces (n x 3) and virials (n x 9) of n atoms, with a full
// neighbor list of at least potential.cutoff in CSR form whose pair vectors
// are those of neighbor::pair_vector; the energies and virials shared among
// the atoms as LAMMPS tallies them, the virials symmetric.
// The potential of its tables: element (17, elements), ebound (elements,
// elements), cmin and cmax (elements^3), tables (pairs, 7, nr); options
// ialloy, mix_ref_t, emb_lin_neg, bkgd_dyn; numbers rc, delr,
// gsmooth_factor, dr, cutoff. Throws std::invalid_argument for cutoffs and
// spacings that are not positive.
MeamPotential make_meam_potential(int elements, const double* element, const double* ebound,
                                  const double* cmin, const double* cmax, const double* tables,
                                  int nr, const int* options, const double* numbers);

void compute_meam(const MeamPotential& potential, std::int64_t n, const std::int32_t* species,
                  const double* positions, const double* matrix, const std::int64_t* offsets,
                  const index_t* neighbors, const std::int8_t* shifts, double* energies,
                  double* forces, double* virials, const Control& control = {},
                  const std::int64_t* reverse = nullptr);

}  // namespace mdapy::calculators

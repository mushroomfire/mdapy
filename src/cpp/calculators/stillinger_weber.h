// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The Stillinger-Weber potential, as LAMMPS pair_style sw computes it
// (Stillinger and Weber, Phys. Rev. B 31, 5262 (1985)).

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

// The parameters of a triplet of elements i, j, k, as a line of a .sw
// file: epsilon sigma a lambda gamma cos(theta0) A B p q tol.
struct StillingerWeberParameters {
    double epsilon = 0, sigma = 0, a = 0, lambda = 0, gamma = 0, costheta = 0, big_a = 0,
           big_b = 0, p = 0, q = 0, tol = 0;
    double cut = 0, cutsq = 0;  // a sigma, and the square of the shorter radius tol gives
    double sigma_gamma = 0, lambda_epsilon = 0, lambda_epsilon2 = 0;
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0;
};

struct StillingerWeberPotential {
    int elements = 0;
    double cutoff = 0.0;  // the largest radius of the pairs
    std::vector<StillingerWeberParameters> parameters;  // (i * elements + j) * elements + k

    const StillingerWeberParameters& of(int i, int j, int k) const {
        return parameters[static_cast<std::size_t>((i * elements + j) * elements + k)];
    }
};

// Builds a potential from rows of 11 numbers, elements^3 of them.
StillingerWeberPotential make_stillinger_weber(int elements, const double* rows);

// Energies (n), forces (n x 3) and virials (n x 9), as compute_tersoff.
void compute_stillinger_weber(const StillingerWeberPotential& potential, std::int64_t n,
                              const std::int32_t* species, const double* positions,
                              const double* matrix, const std::int64_t* offsets,
                              const index_t* neighbors, const std::int8_t* shifts,
                              double* energies, double* forces, double* virials,
                              const Control& control = {});

}  // namespace mdapy::calculators

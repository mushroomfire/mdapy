// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Neuroevolution potentials, version 4: Chebyshev radial functions, real
// spherical harmonics up to l = 8 with optional four- and five-body
// invariants, one hidden layer per element, and an optional ZBL repulsion.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

struct NepModel {
    int types = 0;
    int n_radial = 0, n_angular = 0;  // largest n of the radial functions
    int k_radial = 0, k_angular = 0;  // largest k of the Chebyshev basis
    int l_max = 0;
    bool four_body = false, five_body = false;
    int neurons = 0;
    int dim = 0;
    std::vector<double> rc_radial, rc_angular;  // (types)
    std::vector<double> w0;                     // (types, neurons, dim)
    std::vector<double> b0, w1;                 // (types, neurons)
    double b1 = 0.0;
    std::vector<double> c_radial;   // (types, types, n_radial + 1, k_radial + 1)
    std::vector<double> c_angular;  // (types, types, n_angular + 1, k_angular + 1)
    std::vector<double> q_scaler;   // (dim)
    // ZBL per pair of types, (types, types, 12): inner and outer cutoff,
    // k_e Z_i Z_j, the inverse screening length, four (a, b) of the
    // screening function sum a exp(-b x); empty without ZBL.
    std::vector<double> zbl;
    // Coefficients of the real spherical harmonics, from l = 1, m = 0 by
    // (l, m): the polynomial in z of degree l - m, lowest power first.
    std::vector<double> harmonics;
    // w0 and the coefficients with their last two axes swapped, so that
    // their sums run across neurons and radial functions:
    // (types, dim, neurons), (types, types, k + 1, n + 1).
    std::vector<double> w0_swapped, c_radial_swapped, c_angular_swapped;

    // Fills dim, harmonics and the swapped arrays from the other members.
    void prepare();
    int harmonic_count() const { return (l_max + 1) * (l_max + 1) - 1; }
};

// Energies (n), forces (n x 3), virials (n x 9, may be null) and scaled
// descriptors (n x dim, may be null) of n atoms at positions (n x 3) in a box
// with lattice vectors as rows of matrix (3 x 3), with a full neighbor list in
// CSR form: the neighbor and its periodic image (pairs x 3), whose vectors are
// those of neighbor::pair_vector. With forces null, only the descriptors;
// energies and virials are left untouched. The virial of each derivative of
// the energy of atom i by a neighbor vector goes to that neighbor, as in GPUMD.
void compute_nep(const NepModel& model, std::int64_t n, const std::int32_t* species,
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, double* descriptors,
                 const Control& control = {});

}  // namespace mdapy::calculators

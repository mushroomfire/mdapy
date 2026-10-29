// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Smooth overlap of atomic positions (SOAP): the power spectrum of the
// density of Gaussians on the neighbors of each atom, expanded in Gaussian
// radial functions and real spherical harmonics, in the conventions of
// DScribe.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::neighbor {
struct Frames;
}

namespace mdapy::analysis {

// The expansion: n_max radial functions and degrees up to l_max; the width
// of the Gaussians as eta = 1 / (2 sigma^2); for each l the exponents of
// the Gaussian radial functions (l_max + 1 x n_max) and the matrix that
// makes them orthonormal (l_max + 1 x n_max x n_max, beta[l][n][k]); and
// whether pairs of different species are kept.
struct SoapBasis {
    int n_max = 0;
    int l_max = 0;
    double eta = 0;
    std::vector<double> alphas;
    std::vector<double> betas;
    bool crossover = true;
};

// The number of values per atom for species kinds of atoms.
std::int64_t soap_dimension(const SoapBasis& basis, int species);

// The power spectrum of each of n atoms (n x soap_dimension): species, the
// index of the species of each atom; positions n x 3; matrix, the rows a,
// b, c of the box; the neighbors of each atom in CSR form with their
// periodic images (offsets n + 1, others and shifts 3 each), the atom itself
// left out.
std::vector<double> soap(const SoapBasis& basis, int species_count, const std::int32_t* species,
                         std::int64_t n, const double* positions, const double* matrix,
                         const std::int64_t* offsets, const index_t* others,
                         const std::int8_t* shifts, const Control& control = {});

// The mean power spectrum of each of many frames (frames x soap_dimension),
// each as soap finds it alone with the neighbors closer than cutoff, a frame
// on a thread (species: n atoms in all); NaN for a frame without atoms.
std::vector<double> soap_frames(const SoapBasis& basis, int species_count, const std::int32_t* species,
                                const neighbor::Frames& frames, double cutoff,
                                const Control& control = {});

}  // namespace mdapy::analysis

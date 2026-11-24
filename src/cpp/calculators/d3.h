// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The DFT-D3 dispersion correction of Grimme et al., J. Chem. Phys. 132, 154104
// (2010): C6 from the coordination numbers of the atoms, Becke-Johnson or zero
// damping, and the three-body term of Axilrod, Teller and Muto, as s-dftd3
// computes them, in Å and eV.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::calculators {

enum class D3Damping { Rational, Zero };

// The most reference systems of an element.
constexpr int kD3References = 7;

struct D3Potential {
    D3Damping damping = D3Damping::Rational;
    int species = 0;
    double s6 = 1.0, s8 = 0.0, s9 = 0.0;
    double alpha = 14.0;            // of zero damping; the three-body term takes alpha + 2
    int half_alpha = 7;             // alpha / 2, alpha even
    // C9 = -sqrt(C6 C6 C6) holds in atomic units: in eV Å^9 it is divided by
    // the square root of the hartree in eV
    double hartree = 1.0;
    double cutoff = 0.0;            // of the pairs, in Å
    double cn_cutoff = 0.0;         // of the coordination numbers
    double three_body_cutoff = 0.0; // of the triangles
    double reach = 0.0;             // the largest of the three (two without s9)
    std::vector<int> references;       // species: how many reference systems
    std::vector<double> reference_cn;  // species x kD3References
    std::vector<double> rcov;          // species: 4/3 of the covalent radius, Å
    // species x species
    std::vector<double> ratio;      // C8 / C6 = 3 sqrt(Z_i Z_j) ... in Å^2
    std::vector<double> radius6;    // Becke-Johnson R0^6; zero damping rs6 R0
    std::vector<double> radius8;    // Becke-Johnson R0^8; zero damping rs8 R0
    std::vector<double> atm_radius; // 4/3 R0AB of the three-body damping, Å
    std::vector<std::int64_t> blocks;  // the first C6 of each pair of species
    std::vector<std::int64_t> slots;   // species + 1: the references of the species before
    // for each pair of species, (references of i) x (references of j), eV Å^6
    std::vector<double> c6;
};

// The potential of species elements from their tables: references (species),
// reference_cn (species x kD3References, the first references of each used),
// rcov (species, 4/3 of the covalent radii in Å), r4r2 (species,
// sqrt(0.5 <r^4>/<r^2> sqrt(Z)) in Å), vdw (species x species, R0AB in Å) and
// c6 (for each pair of species, i major, the C6 of their pairs of references
// in eV Å^6, references of i major). a2 in Å, hartree in eV, alpha even. Throws
// std::invalid_argument for tables that do not fit, cutoffs that are not positive
// or an alpha that is not even.
D3Potential make_d3_potential(D3Damping damping, int species, const int* references,
                              const double* reference_cn, const double* rcov,
                              const double* r4r2, const double* vdw, const double* c6,
                              std::int64_t c6_size, double s6, double s8, double s9, double a1,
                              double a2, double rs6, double rs8, double alpha, double hartree,
                              double cutoff, double cn_cutoff, double three_body_cutoff);

// Energies (n), forces (n x 3) and, unless null, virials (n x 9) of n atoms,
// with a full neighbor list in CSR form whose pair vectors are those of
// neighbor::pair_vector and which holds every pair within the reach; pairs
// beyond each cutoff are passed over. Each atom gathers its own terms: half of
// each pair's energy and virial, a third of each triangle's.
void compute_d3(const D3Potential& potential, std::int64_t n, const std::int32_t* species,
                const double* positions, const double* matrix, const std::int64_t* offsets,
                const index_t* neighbors, const std::int8_t* shifts, double* energies,
                double* forces, double* virials, const Control& control = {});

}  // namespace mdapy::calculators

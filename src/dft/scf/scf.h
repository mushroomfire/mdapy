// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The self-consistent field of Kohn and Sham, without spin or with collinear
// spin, on one GPU, in hartree atomic units. Each step takes the input density rho_in, its
// potential V = V_loc + V_H + V_xc, the bands of every k-point by Davidson
// from those of the step before (random at the start), their occupations and
// output density rho_out, and the energy of Harris and Foulkes
//   E = sum_k w_k f_max sum_n f_n e_n - int V_Hxc rho_in + E_H[rho_in]
//       + E_xc[rho_in + rho_core] + E_Ewald - TS,
// the G = 0 term of V_loc (the alpha term) inside the eigenvalues. It stops
// when the Hartree energy of rho_out - rho_in and the change of the energy are
// both below their tolerances; otherwise Pulay's mixing gives the next rho_in.
// With spin, each channel has its bands and potential V_loc + V_H + V_xc,s,
// one Fermi level holds the electrons of both, and the total density and the
// magnetization m = rho_up - rho_down are mixed together.
// At the end, the forces of the bands on the ions: the local term of rho_out,
// that of the projectors, and that of the core charge in V_xc of the last
// rho_in, the derivatives of that energy with the bands fixed.
// The bands are converged to a residual norm of 0.01 sqrt(E_H[rho_out -
// rho_in]) of the step before, 1e-2 at the start and never above that of the
// step before, tightly for the bands that hold electrons and to 1e-2 for the
// others.

#pragma once

#include <vector>

#include "dft/hamiltonian/hamiltonian.h"
#include "dft/lattice/lattice.h"
#include "dft/scf/mixer.h"
#include "dft/scf/occupations.h"
#include "dft/xc/xc.h"

namespace mdapy::dft {

struct ScfInput {
    Lattice lattice;
    Int3 grid{};                 // the density grid, also that of the bands
    double cutoff = 0.0;         // of the bands, hartree
    double density_cutoff = 0.0;  // of the density, hartree
    std::vector<SpeciesProjectors> species;
    std::vector<Atom> atoms;
    std::vector<Vec3> kpoints;  // fractions of the reciprocal vectors
    std::vector<double> weights;  // adding up to one
    // the form factors of each species on the sphere of the density, (species,
    // n): of V_loc, and of rho_core or empty without core charges
    std::vector<double> local_factors, core_factors;
    std::vector<double> density;  // the first rho_in(r) on the grid, up then down with spin
    double ewald = 0.0;
};

struct ScfOptions {
    int spins = 1;  // 1, or 2 for collinear spin
    int bands = 0;  // per k-point and spin
    double electrons = 0.0;
    Functional functional = Functional::Pbe;
    Smearing smearing = Smearing::Gaussian;
    double sigma = 0.0;  // hartree
    MixingOptions mixing;
    double energy_tolerance = 1e-8;   // hartree, change of the energy
    double density_tolerance = 1e-9;  // hartree, E_H[rho_out - rho_in]
    int max_iterations = 100;
    int davidson_iterations = 100;  // per k-point and step
    int davidson_basis = 4;         // the largest basis, times the bands
    double xc_threshold = 1e-10;
    unsigned long long seed = 0;
};

struct ScfStep {
    double energy = 0.0;
    double accuracy = 0.0;  // E_H[rho_out - rho_in], plus the term of m with spin
    double magnetization = 0.0;  // int m of rho_out, Bohr magnetons
    double fermi = 0.0;
    double tolerance = 0.0;  // of the residuals of the bands
    int davidson = 0;        // iterations of Davidson, summed over the k-points
};

struct ScfResult {
    bool converged = false;
    int iterations = 0;
    double energy = 0.0;  // the free energy F
    double band = 0.0, one_electron = 0.0, hartree = 0.0, xc = 0.0, ewald = 0.0, entropy = 0.0;
    double fermi = 0.0, lowest_empty = 0.0;
    double magnetization = 0.0, absolute_magnetization = 0.0;  // int m and int |m|, Bohr magnetons
    std::vector<double> eigenvalues, occupations;  // (spins, k-points, bands)
    std::vector<double> density;                   // the last rho_out(r), (spins, points)
    // (atoms, 3) hartree / bohr: of V_loc, the projectors and the core charge,
    // without the Ewald forces of the ions
    std::vector<double> forces;
    std::vector<ScfStep> steps;
};

ScfResult run_scf(const ScfInput& input, const ScfOptions& options);

}  // namespace mdapy::dft

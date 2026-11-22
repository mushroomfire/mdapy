// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The Kohn-Sham Hamiltonian of norm-conserving pseudopotentials on the plane
// waves of one k-point, applied to a block of bands without a matrix:
//   H psi = |k+G|^2/2 psi + FFT^-1[V_eff(r) FFT[psi]] + sum_ij beta_i D_ij <beta_j|psi>.
// The projectors beta_{I,p,m}(k+G) = (-i)^l Y_lm(k+G) t_p(|k+G|) exp(-i (k+G).tau_I),
// t_p the table of projector p of the species of atom I (radial.projector_table,
// with its 4 pi / sqrt(Omega)), interpolated by Lagrange polynomials of four
// points as QE does, and Y_lm the real spherical harmonics of l up to 3, m
// from -l to l, in the order of the projectors of the species and then of m.
// The overlap S is 1. Hartree atomic units.

#pragma once

#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dft/core/device.h"
#include "dft/fft/fft.h"
#include "dft/lattice/lattice.h"

namespace mdapy::dft {

// The nonlocal part of a species: its projectors tabulated every dq from q =
// 0, their angular momenta and D_ij in hartree.
struct SpeciesProjectors {
    double dq = 0.01;
    int points = 0;                   // of each table
    std::vector<double> tables;       // (projectors, points)
    std::vector<int> angular;         // (projectors)
    std::vector<double> dij;          // (projectors, projectors)
};

struct Atom {
    Vec3 position;  // fractions of the cell vectors
    int species;
};

class Hamiltonian {
public:
    Hamiltonian(const Lattice& lattice, const Int3& grid, double cutoff,
                std::vector<SpeciesProjectors> species, std::vector<Atom> atoms);
    ~Hamiltonian();
    Hamiltonian(const Hamiltonian&) = delete;
    Hamiltonian& operator=(const Hamiltonian&) = delete;

    // The plane waves of k (fractions of the reciprocal vectors) and their
    // projectors.
    void set_k(const Vec3& k);
    // The effective potential V_loc + V_H + V_xc on the grid, real, n1 n2 n3
    // values on the device.
    void set_potential(const double* potential);

    // H on count bands, column-major (plane waves, count), on the device.
    void apply(const Complex* psi, Complex* hpsi, int count);

    const Sphere& sphere() const noexcept { return sphere_; }
    std::size_t plane_waves() const noexcept { return sphere_.size(); }
    int projectors() const noexcept { return projectors_; }
    const DeviceArray<Complex>& beta() const noexcept { return beta_; }
    const DeviceArray<double>& kinetic() const noexcept { return kinetic_; }

private:
    Lattice lattice_;
    Int3 grid_;
    double cutoff_;
    std::vector<SpeciesProjectors> species_;
    std::vector<Atom> atoms_;
    Fft fft_;
    cublasHandle_t blas_ = nullptr;
    int projectors_ = 0;

    // of the k-point
    Sphere sphere_;
    DeviceArray<double> kinetic_;
    DeviceArray<std::int64_t> box_;
    DeviceArray<Complex> beta_;     // (plane waves, projectors)
    DeviceArray<Complex> dmatrix_;  // (projectors, projectors), block diagonal
    // the potential and the work arrays
    DeviceArray<double> potential_;
    DeviceArray<Complex> boxes_, projections_, scaled_;
    // the tables of every species on the device, and where each starts
    DeviceArray<double> tables_;
};

}  // namespace mdapy::dft

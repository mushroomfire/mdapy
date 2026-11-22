// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The augmentation charges of ultrasoft pseudopotentials on the sphere of the
// density, on the GPU. Of the projectors a = (p, m) of an atom, packed in
// pairs a <= b,
//   Q_ab(G) = sum_L (-i)^L Q^L_pq(|G|) sum_M c(l_a m_a, l_b m_b, L M) Y_LM(G),
// Q^L_pq the radial tables (4 pi / Omega) int r^2 Q^L(r) j_L(qr) dr every dq
// (radial.augmentation_tables) interpolated by Lagrange polynomials of four
// points, c the real Gaunt coefficients (angular.gaunt). The augmented density
// and the strengths of the projectors of the atoms I are
//   rho_aug(G) = sum_I sum_{a<=b} becsum^I_ab Q_ab(G) exp(-i G.tau_I),
//   D^I_ab = Omega sum_G Re[conj(V(G)) Q_ab(G) exp(-i G.tau_I)],
// becsum with 2 Re conj(<beta_a|psi>) <beta_b|psi> off the diagonal. The work
// goes by species and in blocks of G, the structure factors through GEMMs; Q is
// never kept for each atom.

#pragma once

#include <cublas_v2.h>

#include <vector>

#include "dft/core/device.h"
#include "dft/hamiltonian/hamiltonian.h"
#include "dft/lattice/lattice.h"

namespace mdapy::dft {

struct SpeciesAugmentation {
    int functions = 0;             // projectors with their m, of one atom
    std::vector<int> pair_start;   // (pairs + 1) the first term of each pair a <= b
    std::vector<int> term_lm;      // the index L^2 + L + M of Y_LM
    std::vector<int> term_l;       // L
    std::vector<int> term_table;   // the table of Q^L_pq
    std::vector<double> term_coef;  // the Gaunt coefficient
    int lmax = 0;                  // the largest L
    double dq = 0.01;
    int points = 0;
    std::vector<double> tables;  // (tables, points)
    int pairs() const noexcept { return static_cast<int>(pair_start.size()) - 1; }
};

class Augmentation {
public:
    Augmentation(const Lattice& lattice, const Sphere& sphere, const std::vector<Atom>& atoms,
                 std::vector<SpeciesAugmentation> species);
    ~Augmentation();
    Augmentation(const Augmentation&) = delete;
    Augmentation& operator=(const Augmentation&) = delete;

    bool any() const noexcept { return any_; }
    int pairs(int species) const { return species_[static_cast<std::size_t>(species)].pairs(); }
    // q_ab = Omega Q_ab(G = 0) of a species, (pairs), on the host
    std::vector<double> overlaps(int species) const;
    // rho(G) on the sphere, on the device, += the augmentation of becsum, on
    // the host: for each atom the pairs of its species, in the order of atoms
    void add_density(const std::vector<double>& becsum, Complex* rho);
    // the strengths Omega sum_G Re[conj(V(G)) Q_ab(G) exp(-i G.tau)] of V(G)
    // on the device, for each atom its pairs, on the host
    std::vector<double> strengths(const Complex* v);
    // Q_ab(G) of one species at an atom at the origin, (pairs, n), on the host;
    // for the tests
    std::vector<Complex> components(int species);

private:
    void block_q(int species, std::size_t first, std::size_t count, Complex* out);
    void block_phases(int species, std::size_t first, std::size_t count, Complex* out);

    Lattice lattice_;
    std::size_t n_;
    std::vector<Atom> atoms_;
    std::vector<SpeciesAugmentation> species_;
    std::vector<std::vector<int>> members_;  // the atoms of each species
    std::vector<std::vector<std::size_t>> offsets_;  // where the pairs of each atom start in becsum
    bool any_ = false;
    cublasHandle_t blas_ = nullptr;
    DeviceArray<double> g_;  // (n, 3) Cartesian G
    DeviceArray<int> miller_;
    struct Device {
        DeviceArray<int> pair_start, term_lm, term_l, term_table;
        DeviceArray<double> term_coef, tables;
        DeviceArray<double> tau;  // the fractional positions of its atoms
    };
    std::vector<Device> device_;
    DeviceArray<Complex> q_, phases_, work_, values_;
};

}  // namespace mdapy::dft

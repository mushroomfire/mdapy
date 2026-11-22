// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The electrostatic energy of the ions, point charges in a uniform background
// that makes the cell neutral, by the Ewald sum, in hartree atomic units:
//   E = 1/2 sum_ij sum'_R z_i z_j erfc(sqrt(eta) |t_ij + R|) / |t_ij + R|
//       + (2 pi / Omega) sum_{G != 0} exp(-G^2 / 4 eta) / G^2 |S(G)|^2
//       - sqrt(eta / pi) sum_i z_i^2 - pi (sum_i z_i)^2 / (2 Omega eta),
// S(G) = sum_i z_i exp(i G.t_i); with the forces and dE/d epsilon of a
// homogeneous strain. eta grows with the cube root of the number of atoms,
// which keeps the two sums of the same work, each cut where its terms fall
// below 1e-16 (as mdapy.calculators._dft.ewald, the reference of the tests).

#pragma once

#include <array>
#include <vector>

namespace mdapy::dft {

struct EwaldResult {
    double energy = 0.0;
    std::vector<double> forces;     // (atoms, 3), hartree / bohr
    std::array<double, 9> strain{};  // dE/d epsilon, (3, 3) row by row, hartree
};

// cell: rows of nine numbers (bohr); positions (atoms, 3) Cartesian (bohr)
EwaldResult ewald(const double* cell, const std::vector<double>& positions, const std::vector<double>& charges);

}  // namespace mdapy::dft

// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The occupations of the bands and the Fermi level, on the host. Fixed
// occupations fill the lowest bands of every k-point. A smearing of width
// sigma occupies a level e by f((e - mu) / sigma) of the step smeared by
//   Gaussian:               delta(u) = exp(-u^2) / sqrt(pi)
//   Fermi-Dirac:            delta(u) = 1 / (4 cosh^2(u / 2))
//   Methfessel-Paxton (1):  delta(u) = exp(-u^2) (3/2 - u^2) / sqrt(pi)
//   Marzari-Vanderbilt:     delta(u) = exp(-s^2) (1 + sqrt(2) s) / sqrt(pi), s = u + 1/sqrt(2)
// with f(u) the integral of delta from u to infinity. The Fermi level mu is
// bracketed and bisected until the bands hold the electrons. The free energy
// F = sum w f e + ... - TS takes the smearing term
//   -TS = sigma sum_k w_k f_max sum_n e(u_n),  e(u) = int_-inf^u t delta(t) dt,
// so that dF/de_n = w_k f_max f_n at a fixed number of electrons.

#pragma once

#include <string>
#include <vector>

namespace mdapy::dft {

enum class Smearing { Fixed, Gaussian, FermiDirac, MethfesselPaxton, MarzariVanderbilt };

Smearing smearing_of(const std::string& name);

struct Occupation {
    std::vector<double> f;  // (k-points, bands), from 0 to 1 (MP and MV a little beyond)
    double fermi = 0.0;     // mu; for fixed occupations the highest occupied level
    double entropy = 0.0;   // -TS, hartree
    double lowest_empty = 0.0;  // for fixed occupations the lowest empty level, else mu
};

// eigenvalues (k-points, bands) row by row and the weights of the k-points,
// adding up to one, or to two over the k-points of both spins; f_max the
// electrons a band holds (2 without spin, 1 with). Fixed occupations fill the
// same bands at every row.
Occupation occupy(const std::vector<double>& eigenvalues, const std::vector<double>& weights, int bands,
                  double electrons, double f_max, Smearing smearing, double sigma);

// f(u) and e(u) of a smearing
double smeared_occupation(Smearing smearing, double u);
double smeared_entropy(Smearing smearing, double u);

}  // namespace mdapy::dft

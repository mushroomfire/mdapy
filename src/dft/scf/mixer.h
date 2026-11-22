// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Pulay's mixing of densities in reciprocal space (Kresse and Furthmueller,
// Comput. Mater. Sci. 6, 15 (1996)), on the host. Each step takes the input
// density of the step and the output of its bands, as Fourier components on
// the sphere of the density, channel by channel (the total density, then the
// magnetization). The residuals R = out - in of the steps kept are combined to
// the least norm in the metric w(G) = (G^2 + q1^2) / G^2 on the total density,
// q1 such that w is metric_weight at the shortest G, and 1 on the
// magnetization:
//   rho_opt = rho_in - sum_i g_i d(rho_in)_i,  R_opt = R - sum_i g_i dR_i,
//   rho_next = rho_opt + P R_opt,
// with Kerker's P(G) = max(minimum, beta G^2 / (G^2 + q0^2)) on the total
// density (beta without q0) and magnetic_beta on the magnetization, in Kerker's
// form with magnetic_kerker but for its G = 0. The
// component G = 0 of the total density holds the electrons and is left as it
// is; that of the magnetization, the total moment, is mixed.

#pragma once

#include <cuComplex.h>

#include <deque>
#include <vector>

namespace mdapy::dft {

struct MixingOptions {
    double beta = 0.7;           // the share of the residual taken
    double kerker = 0.8;         // q0 in 1/bohr; 0 for none
    double minimum = 0.0;        // the least share of any component with Kerker
    double magnetic_beta = 0.4;  // the share of the residual of the magnetization
    double magnetic_kerker = 0.0;  // q0 of Kerker's form on the magnetization but at G = 0, 1/bohr; 0 for none
    int history = 8;             // steps kept
    double metric_weight = 20.0;  // w at the shortest G; 1 for the plain norm
};

class Mixer {
public:
    // g2: |G|^2 of the sphere, 1/bohr^2, G = 0 first; channels 1 or 2
    Mixer(std::vector<double> g2, double volume, int channels, MixingOptions options);

    // The next input density from this step's input and output, (channels, n)
    // each.
    std::vector<cuDoubleComplex> mix(const std::vector<cuDoubleComplex>& in,
                                     const std::vector<cuDoubleComplex>& out);
    // The Hartree energy of the residual of the total density, E_H[out - in],
    // the estimate of the error of the energy; with the magnetization, plus
    // 2 pi Omega sum_G |m_out - m_in|^2 times one bohr^2, the Hartree energy of
    // a residual at the wavelength of one bohr.
    double residual_energy(const std::vector<cuDoubleComplex>& in, const std::vector<cuDoubleComplex>& out) const;
    void reset();

private:
    double dot(const std::vector<cuDoubleComplex>& a, const std::vector<cuDoubleComplex>& b) const;

    std::vector<double> g2_, metric_, precondition_;
    double volume_;
    int channels_;
    MixingOptions options_;
    std::size_t n_;
    std::deque<std::vector<cuDoubleComplex>> inputs_, residuals_;
};

}  // namespace mdapy::dft

// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Pulay's mixing of densities in reciprocal space (Kresse and Furthmueller,
// Comput. Mater. Sci. 6, 15 (1996)), on the GPU. Each step takes the input
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
// Real components outside the metric can come along, as Quantum ESPRESSO
// mixes the occupations of the projectors of PAW: the same g_i, and the share
// beta of their residual, magnetic_beta in the second of two channels.
//
// The densities, the differences of the steps kept and the work stay on the
// GPU; only the small least-squares problem of the g_i and the components that
// come along are on the host. Every sum goes in an order that is always the
// same, so that a run gives the same numbers each time.

#pragma once

#include "dft/core/portable.h"

#include <deque>
#include <functional>
#include <vector>

#include "dft/core/device.h"

namespace mdapy::dft {

struct MixingOptions {
    double beta = 0.7;           // the share of the residual taken
    double kerker = 0.8;         // q0 in 1/bohr; 0 for none
    double minimum = 0.0;        // the least share of any component with Kerker
    double magnetic_beta = 1.6;  // the share of the residual of the magnetization, VASP's AMIX_MAG
    double magnetic_kerker = 0.0;  // q0 of Kerker's form on the magnetization but at G = 0, 1/bohr; 0 for none
    int history = 8;             // steps kept
    double metric_weight = 20.0;  // w at the shortest G; 1 for the plain norm
    // the largest condition number of the residual differences kept: the
    // directions of their least squares with singular values below the
    // largest over this are dropped
    double conditioning = 1e6;
    // the preconditioner of the total density: Kerker's, or the local
    // density of states at the Fermi level (Herbst and Levitt, J. Phys.:
    // Condens. Matter 33, 085503 (2021)) that the SCF sets each step
    bool ldos = false;
};

class Mixer {
public:
    // g2: |G|^2 of the sphere, 1/bohr^2, G = 0 first; channels 1 or 2
    Mixer(std::vector<double> g2, double volume, int channels, MixingOptions options);

    // The next input density from this step's input and output, (channels, n)
    // each on the GPU, into next (which may be in); with extra, the components
    // that come along, on the host: their input, replaced by the next, and
    // their output, (channels, count) each.
    void mix(const cuDoubleComplex* in, const cuDoubleComplex* out, cuDoubleComplex* next,
             std::vector<double>* extra = nullptr, const std::vector<double>* extra_out = nullptr);
    // The Hartree energy of the residual of the total density, E_H[out - in],
    // the estimate of the error of the energy; with the magnetization, plus
    // 2 pi Omega sum_G |m_out - m_in|^2 times one bohr^2, the Hartree energy of
    // a residual at the wavelength of one bohr. in and out on the GPU.
    // With total_only, of the total density alone.
    double residual_energy(const cuDoubleComplex* in, const cuDoubleComplex* out, bool total_only = false);
    void reset();
    // a factor on the shares of the residuals (beta, magnetic_beta and the
    // preconditioner), 1 at the start: the damping of adaptive damping
    void set_scale(double factor);
    double scale() const noexcept { return scale_; }
    // P^-1 on the residual of the total density (its sphere, G = 0 first and
    // zero, on the host), in place, in the place of Kerker's; none for
    // Kerker's again
    void set_preconditioner(std::function<void(std::vector<cuDoubleComplex>&)> apply);

private:
    // <a_i, b> in the metric for the count vectors a on the GPU, into out
    void dots(const std::vector<const cuDoubleComplex*>& a, const cuDoubleComplex* b, double* out);

    std::vector<double> g2_;
    double volume_;
    int channels_;
    MixingOptions options_;
    std::size_t n_;
    double scale_ = 1.0;
    std::function<void(std::vector<cuDoubleComplex>&)> preconditioner_;
    DeviceArray<double> g2_device_, metric_, precondition_;
    // the input and the residual of the last step, the residual of this one,
    // and the work of the combination
    DeviceArray<cuDoubleComplex> last_input_, last_residual_, residual_, combined_, total_;
    bool started_ = false;
    // the differences of the steps kept, d rho_i and d R_i, in slots of which
    // order_ lists the oldest first, with their products,
    // products_[i][j] = <d R_i, d R_j> for j <= i in that order: a step adds
    // one difference and its products with the others instead of all of them
    std::vector<DeviceArray<cuDoubleComplex>> input_slots_, residual_slots_;
    std::deque<int> order_;
    std::deque<std::vector<double>> products_;
    DeviceArray<double> partial_, sums_, coefficients_;
    std::deque<std::vector<double>> extra_inputs_, extra_residuals_;
};

// x of A x = b by GMRES without restart (Saad and Schultz), at most
// iterations steps or a residual below tolerance times |b|
std::vector<cuDoubleComplex> gmres(
    const std::function<std::vector<cuDoubleComplex>(const std::vector<cuDoubleComplex>&)>& apply,
    const std::vector<cuDoubleComplex>& b, int iterations, double tolerance);

}  // namespace mdapy::dft

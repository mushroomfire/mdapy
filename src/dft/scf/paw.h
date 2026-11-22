// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The one-center terms of PAW on the GPU, as Quantum ESPRESSO 7.6 takes them
// (PAW_potential). From the occupations becsum_ab of the pairs a <= b of the
// projector functions of an atom (2 Re off the diagonal), its densities on its
// radial mesh,
//   n_LM(r) = sum_ab becsum_ab G(LM, a, b) phi_a(r) phi_b(r)              (AE)
//   n~_LM(r) = sum_ab becsum_ab G(LM, a, b) (phi~_a phi~_b + r^2 Q^L_ab)(r)  (PS)
// (r^2 times the density, G the Gaunt coefficients of the real harmonics), and
// for each of the two:
//   - the Hartree potential of each LM by Numerov's method on the logarithmic
//     mesh, and E_H = 1/2 sum_LM int v_LM n_LM dr;
//   - XC on the directions of the quadrature (with the core charge), the
//     gradients of GGA from the radial derivative of three points and the
//     angular derivatives of the harmonics, the potential of GGA by the
//     divergence of its flux expanded to two more l;
//   - D_ab = sum_LM G(LM, a, b) int (v_H + v_xc,s)_LM F_ab,L dr up to kkbeta.
// E^1 = E_H + E_xc of AE less PS, and D^1 - D~^1 likewise. Integrals by
// Simpson's rule of the weights given. The atoms of a species go in chunks
// that keep the work arrays near half a gigabyte.

#pragma once

#include <cublas_v2.h>

#include <vector>

#include "dft/core/device.h"
#include "dft/xc/xc.h"

namespace mdapy::dft {

// The one-center data of a species (mdapy.calculators._dft.paw.onecenter);
// mesh 0 for a species without PAW.
struct SpeciesOneCenter {
    int mesh = 0, kkbeta = 0, lmax_rho = 0;
    int l2 = 0, lh = 0, nx = 0;  // (lmax_rho + 1)^2, with the extra l of GGA, directions
    int functions = 0;           // the projector functions w; their pairs w (w + 1) / 2
    int radial = 0;              // the radial pairs Q
    double dx = 0.0;             // of the logarithmic mesh
    std::vector<double> r, whole, inner;          // (mesh): the mesh, Simpson's weights over it and to kkbeta
    std::vector<double> ww, sin_theta, cos_theta;  // (nx)
    std::vector<double> ylm, dylmt, dylmp;        // (nx, lh): Y, dY/dtheta, dY/dphi / sin(theta)
    std::vector<double> ae;                       // (Q, mesh) phi_p phi_q
    std::vector<double> ps;                       // (Q, lmax_rho + 1, mesh) phi~_p phi~_q + r^2 Q^L_pq
    std::vector<double> coef;                     // (pairs, l2) G(LM, a, b)
    std::vector<int> pair;                        // (pairs) the radial pair of each
    std::vector<double> ae_core, ps_core;         // (mesh)
    int pairs() const { return functions * (functions + 1) / 2; }
};

struct OneCenterEnergies {
    double hartree_ae = 0.0, xc_ae = 0.0, hartree_ps = 0.0, xc_ps = 0.0;
    double total() const { return hartree_ae + xc_ae - hartree_ps - xc_ps; }
};

class OneCenter {
public:
    // species by index, mesh 0 without PAW; the species of each atom and the
    // projector functions of each atom, whose pairs pack becsum and D
    OneCenter(std::vector<SpeciesOneCenter> species, std::vector<int> atom_species, std::vector<int> widths,
              int spins, Functional functional, double threshold);
    ~OneCenter();
    OneCenter(const OneCenter&) = delete;
    OneCenter& operator=(const OneCenter&) = delete;

    // E^1 of all atoms and D (each spin, the pairs of all atoms, zero for
    // those without PAW) of the occupations becsum (each spin, likewise)
    double compute(const std::vector<std::vector<double>>& becsum, std::vector<std::vector<double>>& d,
                   OneCenterEnergies* parts = nullptr);

private:
    struct Device {
        DeviceArray<double> r, sqr, whole, inner, ww, sin_theta, ylm, wwylm, dylmt, dylmp, angular, ae, ps, coef;
        DeviceArray<int> pair, ell;
        DeviceArray<double> ae_core, ps_core;
    };
    void chunk(int species, const std::vector<int>& atoms, const std::vector<std::vector<double>>& becsum,
               std::vector<std::vector<double>>& d, OneCenterEnergies& energies);
    void gemm(bool trans_b, int m, int n, int k, const double* a, int lda, long long stride_a, const double* b,
              int ldb, double beta, double* c, int ldc, long long stride_c, int batch);

    std::vector<SpeciesOneCenter> species_;
    std::vector<int> atom_species_;
    std::vector<std::size_t> offsets_;  // where the pairs of each atom start
    int spins_;
    Functional functional_;
    double threshold_;
    bool gga_;
    cublasHandle_t blas_ = nullptr;
    std::vector<Device> device_;
    DeviceArray<double> bec_, cq_, rho_lm_, vh_, scratch_, eh_, n_rad_, gr_, gp_, gt_, rho_xc_, sigma_, f_, v_,
        vs_, exc_, v_rad_, v_lm_, h_, hlm_, div_, div_lm_, integral_, d_;
};

}  // namespace mdapy::dft

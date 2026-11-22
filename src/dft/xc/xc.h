// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Exchange-correlation functionals of collinear spin, in hartree atomic units:
// the energy per volume f(n_up, n_down, s_uu, s_ud, s_dd), s the products of
// the gradients of the spin densities, s_ud = grad n_up . grad n_down, and its
// first derivatives, the potentials of libxc's vrho and vsigma, from automatic
// differentiation:
//   LDA: Slater exchange and the correlation of Perdew and Wang (1992), with
//        the constants of their paper (PW) or libxc's more digits (PW_MOD);
//   PBE: the exchange and correlation of Perdew, Burke and Ernzerhof (1996),
//        on the correlation of PW_MOD, as libxc's GGA_X_PBE and GGA_C_PBE, or
//        on that of PW (PbePw), as Quantum ESPRESSO's PBE; they differ by about
//        1e-6 of the correlation energy.

#pragma once

#include <cmath>
#include <cstddef>

#include "dft/xc/dual.h"

namespace mdapy::dft {

enum class Functional : int { LdaPw = 0, LdaPwMod = 1, Pbe = 2, PbePw = 3 };

namespace xc {

constexpr double kPi = 3.14159265358979323846264338327950288;

struct PwParameters {
    double a, alpha1, beta1, beta2, beta3, beta4;
};

// G(rs) of Perdew and Wang, eq. 10 of their paper, with p = 1
template <class T>
MDAPY_HD T pw_g(const T& rs, const PwParameters& p) {
    using std::log;
    using std::sqrt;
    const T root = sqrt(rs);
    const T denominator =
        2.0 * p.a * (p.beta1 * root + p.beta2 * rs + p.beta3 * rs * root + p.beta4 * rs * rs);
    return -2.0 * p.a * (1.0 + p.alpha1 * rs) * log(1.0 + 1.0 / denominator);
}

// The correlation per particle of the uniform gas at rs and polarization zeta
template <class T>
MDAPY_HD T pw_correlation(const T& rs, const T& zeta, bool modified) {
    using std::pow;
    // the paramagnetic, the ferromagnetic and minus the spin stiffness
    const PwParameters para = modified
        ? PwParameters{0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294}
        : PwParameters{0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294};
    const PwParameters ferro = modified
        ? PwParameters{0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517}
        : PwParameters{0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517};
    const PwParameters stiff = modified
        ? PwParameters{0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671}
        : PwParameters{0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671};
    const double fz20 = modified ? 1.709920934161365617563962776245 : 1.709921;
    const T e0 = pw_g(rs, para), e1 = pw_g(rs, ferro), alpha = -pw_g(rs, stiff);
    const double two43 = 2.5198420997897463295344212145565;  // 2^(4/3)
    const T fz = (pow(1.0 + zeta, 4.0 / 3.0) + pow(1.0 - zeta, 4.0 / 3.0) - 2.0) / (two43 - 2.0);
    const T z4 = zeta * zeta * zeta * zeta;
    return e0 + alpha * fz / fz20 * (1.0 - z4) + (e1 - e0) * fz * z4;
}

// Slater exchange per volume of one spin density n, as of the density 2n of a
// paramagnetic gas, halved: -(3/4)(3/pi)^(1/3) 2^(1/3) n^(4/3)
template <class T>
MDAPY_HD T slater_spin(const T& n) {
    using std::cbrt;
    return -0.75 * std::cbrt(3.0 / kPi) * std::cbrt(2.0) * n * cbrt(n);
}

// PBE exchange per volume of one spin density n with |grad n|^2 = s, from
// that of the paramagnetic gas of density 2n and gradient squared 4s, halved
template <class T>
MDAPY_HD T pbe_exchange_spin(const T& n, const T& s) {
    using std::cbrt;
    const double kappa = 0.804, mu = 0.2195149727645171;
    const T n2 = 2.0 * n;
    const T kf = std::cbrt(3.0 * kPi * kPi) * cbrt(n2);
    const T s2 = 4.0 * s / (4.0 * kf * kf * n2 * n2);
    const T fx = 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa);
    return 0.5 * (-0.75 * std::cbrt(3.0 / kPi) * n2 * cbrt(n2)) * fx;
}

// The PBE gradient correction of correlation per particle, H(rs, zeta, t)
template <class T>
MDAPY_HD T pbe_h(const T& n, const T& zeta, const T& grad2, const T& ec) {
    using std::cbrt;
    using std::exp;
    using std::log;
    using std::pow;
    using std::sqrt;
    const double beta = 0.06672455060314922;
    const double gamma = (1.0 - 0.69314718055994530941723212145818) / (kPi * kPi);
    const T phi = 0.5 * (pow(1.0 + zeta, 2.0 / 3.0) + pow(1.0 - zeta, 2.0 / 3.0));
    const T kf = std::cbrt(3.0 * kPi * kPi) * cbrt(n);
    const T ks2 = 4.0 * kf / kPi;
    const T t2 = grad2 / (4.0 * phi * phi * ks2 * n * n);
    const T phi3 = phi * phi * phi;
    const T a = (beta / gamma) / (exp(-ec / (gamma * phi3)) - 1.0);
    const T at2 = a * t2;
    return gamma * phi3 *
           log(1.0 + (beta / gamma) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2));
}

// The energy per volume of a functional at one point
template <class T>
MDAPY_HD T energy(Functional functional, const T& up, const T& down, const T& suu, const T& sud,
                  const T& sdd) {
    using std::cbrt;
    const T n = up + down;
    T zeta = (up - down) / n;
    // a fully polarized point stays a hair inside, where the powers are smooth
    const double edge = 1.0 - 1e-12;
    if (value_of(zeta) > edge) zeta = T(edge);
    if (value_of(zeta) < -edge) zeta = T(-edge);
    const T rs = std::cbrt(3.0 / (4.0 * kPi)) / cbrt(n);
    if (functional == Functional::Pbe || functional == Functional::PbePw) {
        const T ec = pw_correlation(rs, zeta, functional == Functional::Pbe);
        const T grad2 = suu + 2.0 * sud + sdd;
        return pbe_exchange_spin(up, suu) + pbe_exchange_spin(down, sdd) +
               n * (ec + pbe_h(n, zeta, grad2, ec));
    }
    const T ec = pw_correlation(rs, zeta, functional == Functional::LdaPwMod);
    return slater_spin(up) + slater_spin(down) + n * ec;
}

}  // namespace xc

// The functional at each of count points: f the energy per volume, v (2 per
// point) its derivatives by n_up and n_down, vs (3 per point) by s_uu, s_ud
// and s_dd. Points of total density below threshold, or a spin density not
// above zero, give zero.
void evaluate_xc(Functional functional, std::size_t count, const double* rho, const double* sigma,
                 double threshold, double* f, double* v, double* vs);

}  // namespace mdapy::dft

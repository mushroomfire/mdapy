// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Real spherical harmonics of l up to 6, of any number type (the duals of the
// stress too), in the convention of the projectors and of
// mdapy.calculators._dft.angular: Y_lm at l^2 + l + m, m from -l to l, the
// sines for m < 0 and the cosines for m > 0, P_l^m without the phase of
// Condon and Shortley. The zero vector has Y_00 alone.

#pragma once

#include <cmath>

#include "dft/xc/dual.h"

namespace mdapy::dft {

constexpr int kHarmonicsMax = 6;

// out[(lmax + 1)^2] the Y_lm of the direction of (x, y, z)
template <class T>
MDAPY_HD void real_harmonics(int lmax, T x, T y, T z, T* out) {
    using std::sqrt;
    const double pi = 3.14159265358979323846264338327950288;
    const T r2 = x * x + y * y + z * z;
    if (value_of(r2) < 1e-24) {
        out[0] = T(0.5 / sqrt(pi));
        for (int i = 1; i < (lmax + 1) * (lmax + 1); ++i) out[i] = T(0.0);
        return;
    }
    const T r = sqrt(r2);
    x = x / r;
    y = y / r;
    z = z / r;
    // cos(m p) rho^m and sin(m p) rho^m as polynomials of x and y
    T c[kHarmonicsMax + 1], s[kHarmonicsMax + 1];
    c[0] = T(1.0);
    s[0] = T(0.0);
    for (int m = 1; m <= lmax; ++m) {
        c[m] = x * c[m - 1] - y * s[m - 1];
        s[m] = y * c[m - 1] + x * s[m - 1];
    }
    // P_l^m(z) / rho^m by recursion: q[l][m]
    T q[kHarmonicsMax + 1][kHarmonicsMax + 1];
    double dfact = 1.0;  // (2m - 1)!!
    for (int m = 0; m <= lmax; ++m) {
        if (m > 0) dfact *= 2.0 * m - 1.0;
        q[m][m] = T(dfact);
        if (m + 1 <= lmax) q[m + 1][m] = z * (2.0 * m + 1.0) * q[m][m];
        for (int l = m + 2; l <= lmax; ++l)
            q[l][m] = ((2.0 * l - 1.0) * z * q[l - 1][m] - (l + m - 1.0) * q[l - 2][m]) / (l - m + 0.0);
    }
    for (int l = 0; l <= lmax; ++l) {
        double ratio = 1.0;  // (l - m)! / (l + m)!
        for (int m = 0; m <= l; ++m) {
            if (m > 0) ratio /= (l - m + 1.0) * (l + m + 0.0);
            const double norm = sqrt((2.0 * l + 1.0) / (4.0 * pi) * ratio);
            if (m == 0) {
                out[l * l + l] = norm * q[l][0];
            } else {
                const T base = sqrt(2.0) * norm * q[l][m];
                out[l * l + l + m] = base * c[m];
                out[l * l + l - m] = base * s[m];
            }
        }
    }
}

}  // namespace mdapy::dft

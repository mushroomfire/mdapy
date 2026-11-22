// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Numbers of forward automatic differentiation: a value and its derivatives
// with respect to N inputs, carried through the arithmetic and the functions
// the exchange-correlation functionals use. A functional written once for a
// scalar type gives its value as double and its exact first derivatives as
// Dual<N>, on the host and the GPU.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define MDAPY_HD __host__ __device__ inline
#else
#define MDAPY_HD inline
#endif

namespace mdapy::dft {

template <int N>
struct Dual {
    double v = 0.0;
    double d[N] = {};

    MDAPY_HD Dual() = default;
    MDAPY_HD Dual(double value) : v(value) {}  // NOLINT: constants mix with duals

    // The input i of the N, of value x.
    MDAPY_HD static Dual input(double x, int i) {
        Dual out(x);
        out.d[i] = 1.0;
        return out;
    }
};

template <int N>
MDAPY_HD Dual<N> chain(const Dual<N>& a, double value, double slope) {
    Dual<N> out(value);
    for (int i = 0; i < N; ++i) out.d[i] = slope * a.d[i];
    return out;
}

template <int N>
MDAPY_HD Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> out(a.v + b.v);
    for (int i = 0; i < N; ++i) out.d[i] = a.d[i] + b.d[i];
    return out;
}
template <int N>
MDAPY_HD Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> out(a.v - b.v);
    for (int i = 0; i < N; ++i) out.d[i] = a.d[i] - b.d[i];
    return out;
}
template <int N>
MDAPY_HD Dual<N> operator-(const Dual<N>& a) {
    return chain(a, -a.v, -1.0);
}
template <int N>
MDAPY_HD Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> out(a.v * b.v);
    for (int i = 0; i < N; ++i) out.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return out;
}
template <int N>
MDAPY_HD Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    const double q = a.v / b.v;
    Dual<N> out(q);
    for (int i = 0; i < N; ++i) out.d[i] = (a.d[i] - q * b.d[i]) / b.v;
    return out;
}
template <int N> MDAPY_HD Dual<N> operator+(const Dual<N>& a, double b) { return a + Dual<N>(b); }
template <int N> MDAPY_HD Dual<N> operator+(double a, const Dual<N>& b) { return Dual<N>(a) + b; }
template <int N> MDAPY_HD Dual<N> operator-(const Dual<N>& a, double b) { return a - Dual<N>(b); }
template <int N> MDAPY_HD Dual<N> operator-(double a, const Dual<N>& b) { return Dual<N>(a) - b; }
template <int N> MDAPY_HD Dual<N> operator*(const Dual<N>& a, double b) { return chain(a, a.v * b, b); }
template <int N> MDAPY_HD Dual<N> operator*(double a, const Dual<N>& b) { return chain(b, a * b.v, a); }
template <int N> MDAPY_HD Dual<N> operator/(const Dual<N>& a, double b) { return chain(a, a.v / b, 1.0 / b); }
template <int N> MDAPY_HD Dual<N> operator/(double a, const Dual<N>& b) { return Dual<N>(a) / b; }

template <int N>
MDAPY_HD Dual<N> log(const Dual<N>& a) {
    return chain(a, std::log(a.v), 1.0 / a.v);
}
template <int N>
MDAPY_HD Dual<N> exp(const Dual<N>& a) {
    const double e = std::exp(a.v);
    return chain(a, e, e);
}
template <int N>
MDAPY_HD Dual<N> sqrt(const Dual<N>& a) {
    const double s = std::sqrt(a.v);
    return chain(a, s, 0.5 / s);
}
template <int N>
MDAPY_HD Dual<N> cbrt(const Dual<N>& a) {
    const double c = std::cbrt(a.v);
    return chain(a, c, c / (3.0 * a.v));
}
// a^p for a > 0
template <int N>
MDAPY_HD Dual<N> pow(const Dual<N>& a, double p) {
    const double value = std::pow(a.v, p);
    return chain(a, value, p * value / a.v);
}

MDAPY_HD double value_of(double x) { return x; }
template <int N>
MDAPY_HD double value_of(const Dual<N>& x) {
    return x.v;
}

}  // namespace mdapy::dft

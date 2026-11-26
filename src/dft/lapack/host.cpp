// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/lapack/host.h"

#include "dft/lapack/divide.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mdapy::dft::host {

namespace {

double conj_of(double x) { return x; }
std::complex<double> conj_of(const std::complex<double>& x) { return std::conj(x); }
double real_of(double x) { return x; }
double real_of(const std::complex<double>& x) { return x.real(); }
double imag_of(double) { return 0.0; }
double imag_of(const std::complex<double>& x) { return x.imag(); }
double norm_of(double x) { return x * x; }
double norm_of(const std::complex<double>& x) { return std::norm(x); }

// a = q t q^H: on return d (n) and e (n - 1) are the diagonal and the
// subdiagonal of the real tridiagonal t, and a holds q. The reflectors are
// LAPACK's: h = 1 - tau v v^H with v_0 = 1 takes the column below the diagonal
// to a real multiple of the first unit vector, h^H x = beta e_1.
template <class S>
void tridiagonalize(S* a, int n, std::vector<double>& d, std::vector<double>& e) {
    const std::size_t ld = static_cast<std::size_t>(n);
    auto at = [&](int i, int j) -> S& { return a[static_cast<std::size_t>(j) * ld + i]; };
    // the lower triangle from the upper one; the diagonal is real
    for (int j = 0; j < n; ++j) {
        at(j, j) = real_of(at(j, j));
        for (int i = j + 1; i < n; ++i) at(i, j) = conj_of(at(j, i));
    }
    std::vector<S> tau(static_cast<std::size_t>(std::max(n - 1, 0))), p(ld), w(ld);
    for (int k = 0; k + 1 < n; ++k) {
        // the reflector of a[k + 1 :, k]
        const S alpha = at(k + 1, k);
        double tail = 0.0;
        for (int i = k + 2; i < n; ++i) tail += norm_of(at(i, k));
        if (tail == 0.0 && imag_of(alpha) == 0.0) {
            tau[k] = S(0.0);
            e[k] = real_of(alpha);
            d[k] = real_of(at(k, k));
            at(k + 1, k) = S(1.0);
            continue;
        }
        const double beta = -std::copysign(std::sqrt(norm_of(alpha) + tail), real_of(alpha));
        tau[k] = (S(beta) - alpha) / beta;
        const S scale = S(1.0) / (alpha - S(beta));
        at(k + 1, k) = S(1.0);
        for (int i = k + 2; i < n; ++i) at(i, k) *= scale;
        e[k] = beta;
        d[k] = real_of(at(k, k));
        // b <- h^H b h of the trailing block b: p = tau b v,
        // w = p - (tau / 2) (p^H v) v, b <- b - v w^H - w v^H
        const int first = k + 1;
        for (int i = first; i < n; ++i) p[i] = S(0.0);
        for (int j = first; j < n; ++j) {
            const S vj = at(j, k);
            for (int i = first; i < n; ++i) p[i] += at(i, j) * vj;
        }
        S dot = S(0.0);
        for (int i = first; i < n; ++i) {
            p[i] *= tau[k];
            dot += conj_of(p[i]) * at(i, k);
        }
        const S half = S(-0.5) * tau[k] * dot;
        for (int i = first; i < n; ++i) w[i] = p[i] + half * at(i, k);
        for (int j = first; j < n; ++j) {
            const S cw = conj_of(w[j]), cv = conj_of(at(j, k));
            for (int i = first; i < n; ++i) at(i, j) -= at(i, k) * cw + w[i] * cv;
        }
    }
    if (n > 0) d[n - 1] = real_of(at(n - 1, n - 1));
    // q = h_0 h_1 ... from the last reflector to the first; reflector k acts
    // on the indices beyond k and is kept in column k below the diagonal
    std::vector<S> v(ld);
    for (int k = n - 2; k >= 0; --k) {
        for (int i = k + 1; i < n; ++i) v[i] = at(i, k);
        // column and row k + 1 of q, and column k, are those of the identity so far
        for (int i = 0; i < n; ++i) at(i, k) = S(0.0);
        at(k, k) = S(1.0);
        if (k == n - 2) {
            for (int i = 0; i < n; ++i) at(i, n - 1) = S(0.0);
            at(n - 1, n - 1) = S(1.0);
        }
        if (tau[k] == S(0.0)) continue;
        for (int j = k + 1; j < n; ++j) {
            S dot = S(0.0);
            for (int i = k + 1; i < n; ++i) dot += conj_of(v[i]) * at(i, j);
            dot *= tau[k];
            for (int i = k + 1; i < n; ++i) at(i, j) -= v[i] * dot;
        }
    }
    if (n == 1) at(0, 0) = S(1.0);
}

// The implicit QL iteration (EISPACK's tql2) on the tridiagonal d, e, the
// rotations applied to the columns of z (n, n); d ascending on return.
template <class S>
void ql(std::vector<double>& d, std::vector<double>& e, S* z, int n) {
    const std::size_t ld = static_cast<std::size_t>(n);
    const double eps = std::numeric_limits<double>::epsilon();
    e.resize(ld);
    if (n > 0) e[n - 1] = 0.0;
    double f = 0.0, tst1 = 0.0;
    for (int l = 0; l < n; ++l) {
        tst1 = std::max(tst1, std::abs(d[l]) + std::abs(e[l]));
        int m = l;
        while (m < n - 1 && std::abs(e[m]) > eps * tst1) ++m;
        if (m > l) {
            int iterations = 0;
            do {
                if (++iterations > 60) throw std::runtime_error("the eigenvalues of a subspace did not converge");
                double g = d[l];
                double p = (d[l + 1] - g) / (2.0 * e[l]);
                double r = std::hypot(p, 1.0);
                if (p < 0.0) r = -r;
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                const double dl1 = d[l + 1];
                double h = g - d[l];
                for (int i = l + 2; i < n; ++i) d[i] -= h;
                f += h;
                p = d[m];
                double c = 1.0, c2 = c, c3 = c, s = 0.0, s2 = 0.0;
                const double el1 = e[l + 1];
                for (int i = m - 1; i >= l; --i) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = std::hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    S* zi = z + static_cast<std::size_t>(i) * ld;
                    S* zj = zi + ld;
                    for (int k = 0; k < n; ++k) {
                        const S t = zj[k];
                        zj[k] = s * zi[k] + c * t;
                        zi[k] = c * zi[k] - s * t;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
            } while (std::abs(e[l]) > eps * tst1);
        }
        d[l] += f;
        e[l] = 0.0;
    }
    // ascending, the vectors with their values
    std::vector<int> order(ld);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return d[i] < d[j]; });
    std::vector<double> sorted(ld);
    std::vector<S> columns(ld * ld);
    for (int j = 0; j < n; ++j) {
        sorted[j] = d[order[j]];
        std::copy(z + static_cast<std::size_t>(order[j]) * ld, z + static_cast<std::size_t>(order[j] + 1) * ld,
                  columns.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(j) * ld));
    }
    d = sorted;
    std::copy(columns.begin(), columns.end(), z);
}

template <class S>
void eigh_of(S* a, int n, double* w) {
    if (n < 1) return;
    std::vector<double> d(static_cast<std::size_t>(n)), e(static_cast<std::size_t>(n));
    tridiagonalize(a, n, d, e);
    ql(d, e, a, n);
    std::copy(d.begin(), d.end(), w);
}

template <class S>
int potrf_of(S* a, int n) {
    const std::size_t ld = static_cast<std::size_t>(n);
    auto at = [&](int i, int j) -> S& { return a[static_cast<std::size_t>(j) * ld + i]; };
    for (int j = 0; j < n; ++j) {
        double pivot = real_of(at(j, j));
        for (int k = 0; k < j; ++k) pivot -= norm_of(at(k, j));
        if (!(pivot > 0.0)) return j + 1;  // also when it is not a number
        const double r = std::sqrt(pivot);
        at(j, j) = r;
        for (int i = j + 1; i < n; ++i) {
            S value = at(j, i);
            for (int k = 0; k < j; ++k) value -= conj_of(at(k, j)) * at(k, i);
            at(j, i) = value / r;
        }
    }
    return 0;
}

}  // namespace

void tridiagonal_ql(std::vector<double>& d, std::vector<double>& e, std::vector<double>& z) {
    const int n = static_cast<int>(d.size());
    z.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) z[static_cast<std::size_t>(i) * n + i] = 1.0;
    e.resize(d.size());
    ql(d, e, z.data(), n);
}

void eigh(std::complex<double>* a, int n, double* w) { eigh_of(a, n, w); }
void eigh(double* a, int n, double* w) { eigh_of(a, n, w); }
int potrf(std::complex<double>* a, int n) { return potrf_of(a, n); }
int potrf(double* a, int n) { return potrf_of(a, n); }

}  // namespace mdapy::dft::host

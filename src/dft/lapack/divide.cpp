// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/lapack/divide.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mdapy::dft::host {

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Root i (of k) of 1 + rho sum_j w_j^2 / (d_j - x) = 0, rho > 0 and d
// ascending and distinct: the one between d_i and d_(i + 1), or beyond d_(k - 1)
// for the last. delta (k) on return holds d_j - x, each from the nearer pole so
// that it is accurate; the root itself is returned.
double secular_root(int i, int k, const double* d, const double* w2, double rho, double* delta) {
    const bool last = i == k - 1;
    double norm = 0.0;
    for (int j = 0; j < k; ++j) norm += w2[j];
    const double gap = last ? rho * norm : d[i + 1] - d[i];
    // the sign of the function at the middle chooses the pole the root is nearer to
    auto value_from = [&](int origin, double mu, double* slope_low, double* slope_high, double* low, double* high,
                          double* size) {
        // psi over the poles up to i, phi over those beyond: f = 1 + rho (psi + phi)
        double psi = 0.0, dpsi = 0.0, phi = 0.0, dphi = 0.0, magnitude = 0.0;
        for (int j = 0; j < k; ++j) {
            const double dj = (d[j] - d[origin]) - mu;
            delta[j] = dj;
            const double t = w2[j] / dj;
            magnitude += std::abs(t);
            if (j <= i) {
                psi += t;
                dpsi += t / dj;
            } else {
                phi += t;
                dphi += t / dj;
            }
        }
        *low = psi;
        *high = phi;
        *slope_low = dpsi;
        *slope_high = dphi;
        *size = 1.0 + rho * magnitude;
        return 1.0 + rho * (psi + phi);
    };
    double a, b, c, e, size;
    int origin = i;
    if (!last) {
        const double middle = value_from(i, 0.5 * gap, &a, &b, &c, &e, &size);
        if (middle < 0.0) origin = i + 1;
    }
    // the root as d_origin + mu, mu within (lo, hi)
    double lo = origin == i ? 0.0 : -gap, hi = origin == i ? gap : 0.0;
    if (!last) {
        if (origin == i)
            hi = 0.5 * gap;
        else
            lo = -0.5 * gap;
    }
    double mu = 0.5 * (lo + hi);
    for (int iteration = 0; iteration < 200; ++iteration) {
        double dpsi, dphi, psi, phi;
        const double f = value_from(origin, mu, &dpsi, &dphi, &psi, &phi, &size);
        if (std::abs(f) <= 8.0 * kEps * size) break;
        if (f < 0.0)
            lo = mu;
        else
            hi = mu;
        if (hi - lo <= 2.0 * kEps * std::max(std::abs(lo), std::abs(hi))) {
            mu = 0.5 * (lo + hi);
            value_from(origin, mu, &dpsi, &dphi, &psi, &phi, &size);
            break;
        }
        // psi and phi each as a constant plus a simple pole, at d_i and at
        // d_(i + 1), with their values and slopes here: the root of the
        // resulting quadratic within the bracket, or its middle
        const double di = delta[i];                           // d_i - x
        const double dn = last ? 0.0 : delta[i + 1];          // d_(i + 1) - x
        const double p = dpsi * di * di, s = psi - dpsi * di;  // psi ~ s + p / (d_i - x)
        const double q = last ? 0.0 : dphi * dn * dn, r = last ? 0.0 : phi - dphi * dn;
        const double constant = 1.0 + rho * (s + r);
        double next = 0.5 * (lo + hi);
        bool found = false;
        if (last || q == 0.0) {
            // constant + rho p / (d_i - x) = 0: d_i - x = -rho p / constant
            if (constant > 0.0) {
                const double step = di + rho * p / constant;  // (d_i - x) - (d_i - x_new)
                next = mu + step;
                found = true;
            }
        } else {
            // constant (di - t)(dn - t) + rho p (dn - t) + rho q (di - t) = 0 in the step t = x_new - x
            const double qa = constant;
            const double qb = -(constant * (di + dn) + rho * (p + q));
            const double qc = constant * di * dn + rho * (p * dn + q * di);
            const double disc = qb * qb - 4.0 * qa * qc;
            if (disc >= 0.0) {
                const double root = std::sqrt(disc);
                const double big = qb >= 0.0 ? -(qb + root) / 2.0 : -(qb - root) / 2.0;
                const double t1 = qa != 0.0 ? big / qa : qc / (-qb), t2 = big != 0.0 ? qc / big : t1;
                for (const double t : {t1, t2})
                    if (mu + t > lo && mu + t < hi && !found) {
                        next = mu + t;
                        found = true;
                    }
            }
        }
        if (!found || !(next > lo && next < hi)) next = 0.5 * (lo + hi);
        mu = next;
    }
    return d[origin] + mu;
}

struct Problem {
    std::vector<double>& d;
    std::vector<double>& e;
    Vectors& q;
    int leaf;
};

void solve(Problem& t, int offset, int size) {
    double* d = t.d.data() + offset;
    if (size <= t.leaf) {
        std::vector<double> dd(d, d + size), ee(t.e.begin() + offset, t.e.begin() + offset + std::max(size - 1, 0)), z;
        tridiagonal_ql(dd, ee, z);
        std::copy(dd.begin(), dd.end(), d);
        t.q.set(offset, size, z.data());
        return;
    }
    const int half = size / 2, middle = offset + half;
    // t = (t_1, t_2) + |rho| u u^T with u the last unit vector of the first
    // half and, with the sign of rho, the first of the second
    const double coupling = t.e[middle - 1], rho0 = std::abs(coupling), sign = coupling < 0.0 ? -1.0 : 1.0;
    t.d[middle - 1] -= rho0;
    t.d[middle] -= rho0;
    solve(t, offset, half);
    solve(t, middle, size - half);
    const std::size_t n = static_cast<std::size_t>(size);
    std::vector<double> z(n), second(n);
    t.q.rows(offset, size, middle, z.data(), second.data());
    for (int j = half; j < size; ++j) z[j] = sign * second[j];
    // z to unit length: its two rows are unit vectors
    double length = 0.0;
    for (const double x : z) length += x * x;
    length = std::sqrt(length);
    for (double& x : z) x /= length;
    const double rho = rho0 * length * length;
    // the poles ascending
    std::vector<int> by(n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return d[a] < d[b]; });
    double dmax = 0.0, zmax = 0.0;
    for (int j = 0; j < size; ++j) {
        dmax = std::max(dmax, std::abs(d[j]));
        zmax = std::max(zmax, std::abs(z[j]));
    }
    const double tolerance = 8.0 * kEps * std::max(dmax, zmax);
    std::vector<int> kept, deflated;
    std::vector<double> values(n);  // of the columns, the deflated ones final
    if (rho * zmax <= tolerance) {
        // nothing couples the halves
        for (const int j : by) deflated.push_back(j);
        for (int j = 0; j < size; ++j) values[j] = d[j];
    } else {
        int candidate = -1;
        for (const int j : by) {
            if (rho * std::abs(z[j]) <= tolerance) {
                deflated.push_back(j);
                values[j] = d[j];
                continue;
            }
            if (candidate < 0) {
                candidate = j;
                continue;
            }
            // two poles within the tolerance of each other: a rotation leaves
            // the first without a component
            const double s0 = z[candidate], c0 = z[j], tau = std::hypot(c0, s0), gap = d[j] - d[candidate];
            const double c = c0 / tau, s = -s0 / tau;
            if (std::abs(gap * c * s) <= tolerance) {
                z[j] = tau;
                z[candidate] = 0.0;
                t.q.rotate(offset, size, candidate, j, c, s);
                const double first = d[candidate] * c * c + d[j] * s * s;
                d[j] = d[candidate] * s * s + d[j] * c * c;
                d[candidate] = first;
                deflated.push_back(candidate);
                values[candidate] = first;
            } else {
                kept.push_back(candidate);
            }
            candidate = j;
        }
        if (candidate >= 0) kept.push_back(candidate);
    }
    const int k = static_cast<int>(kept.size());
    std::vector<double> u(static_cast<std::size_t>(k) * k);
    std::vector<double> roots(static_cast<std::size_t>(k));
    if (k > 0) {
        std::vector<double> pole(static_cast<std::size_t>(k)), w(static_cast<std::size_t>(k)), w2(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) {
            pole[j] = d[kept[j]];
            w[j] = z[kept[j]];
            w2[j] = w[j] * w[j];
        }
        // column i of u holds d_j - root_i
        for (int i = 0; i < k; ++i) roots[i] = secular_root(i, k, pole.data(), w2.data(), rho, u.data() + static_cast<std::size_t>(i) * k);
        // w from the roots: w_j^2 = prod_i (root_i - d_j) / (rho prod_(i != j) (d_i - d_j))
        std::vector<double> fitted(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) {
            double product = -u[static_cast<std::size_t>(j) * k + j] / rho;
            for (int i = 0; i < k; ++i)
                if (i != j) product *= u[static_cast<std::size_t>(i) * k + j] / (pole[j] - pole[i]);
            fitted[j] = std::copysign(std::sqrt(std::abs(product)), w[j]);
        }
        for (int i = 0; i < k; ++i) {
            double* column = u.data() + static_cast<std::size_t>(i) * k;
            double norm = 0.0;
            for (int j = 0; j < k; ++j) {
                column[j] = fitted[j] / column[j];
                norm += column[j] * column[j];
            }
            norm = std::sqrt(norm);
            for (int j = 0; j < k; ++j) column[j] /= norm;
        }
    }
    // the columns [kept u, deflated] ascending by their values
    std::vector<double> all(n);
    for (int i = 0; i < k; ++i) all[i] = roots[i];
    for (std::size_t i = 0; i < deflated.size(); ++i) all[static_cast<std::size_t>(k) + i] = values[deflated[i]];
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return all[a] < all[b]; });
    t.q.merge(offset, size, kept, u.data(), deflated, order);
    for (int j = 0; j < size; ++j) d[j] = all[order[j]];
}

}  // namespace

void divide_and_conquer(std::vector<double>& d, std::vector<double>& e, Vectors& q, int leaf) {
    const int n = static_cast<int>(d.size());
    if (n < 1) return;
    e.resize(d.size());
    // scaled to a largest element of one, so that the tolerances are relative
    double scale = 0.0;
    for (int i = 0; i < n; ++i) scale = std::max({scale, std::abs(d[i]), i + 1 < n ? std::abs(e[i]) : 0.0});
    if (scale == 0.0) scale = 1.0;
    for (int i = 0; i < n; ++i) {
        d[i] /= scale;
        e[i] /= scale;
    }
    Problem t{d, e, q, std::max(leaf, 2)};
    solve(t, 0, n);
    for (double& x : d) x *= scale;
}

void HostVectors::set(int offset, int size, const double* z) {
    for (int j = 0; j < size; ++j)
        for (int i = 0; i < size; ++i)
            q_[static_cast<std::size_t>(offset + j) * n_ + offset + i] = z[static_cast<std::size_t>(j) * size + i];
}

void HostVectors::rows(int offset, int size, int middle, double* a, double* b) {
    for (int j = 0; j < size; ++j) {
        const std::size_t column = static_cast<std::size_t>(offset + j) * n_;
        a[j] = q_[column + middle - 1];
        b[j] = q_[column + middle];
    }
}

void HostVectors::rotate(int offset, int size, int i, int j, double c, double s) {
    double* x = q_.data() + static_cast<std::size_t>(offset + i) * n_ + offset;
    double* y = q_.data() + static_cast<std::size_t>(offset + j) * n_ + offset;
    for (int r = 0; r < size; ++r) {
        const double a = x[r], b = y[r];
        x[r] = c * a + s * b;
        y[r] = c * b - s * a;
    }
}

void HostVectors::merge(int offset, int size, const std::vector<int>& kept, const double* u,
                        const std::vector<int>& deflated, const std::vector<int>& order) {
    const std::size_t n = static_cast<std::size_t>(size), k = kept.size();
    std::vector<double> next(n * n, 0.0);
    auto column = [&](int j) { return q_.data() + static_cast<std::size_t>(offset + j) * n_ + offset; };
    for (std::size_t i = 0; i < k; ++i) {
        double* out = next.data() + i * n;
        for (std::size_t j = 0; j < k; ++j) {
            const double weight = u[i * k + j];
            const double* from = column(kept[j]);
            for (std::size_t r = 0; r < n; ++r) out[r] += weight * from[r];
        }
    }
    for (std::size_t i = 0; i < deflated.size(); ++i) std::copy(column(deflated[i]), column(deflated[i]) + n, next.data() + (k + i) * n);
    for (std::size_t j = 0; j < n; ++j)
        std::copy(next.data() + static_cast<std::size_t>(order[j]) * n, next.data() + static_cast<std::size_t>(order[j] + 1) * n, column(static_cast<int>(j)));
}

}  // namespace mdapy::dft::host

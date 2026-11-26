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
            const double dj = (d[j] - d[origin]) - mu, inverse = 1.0 / dj;
            delta[j] = dj;
            const double t = w2[j] * inverse;
            magnitude += std::abs(t);
            if (j <= i) {
                psi += t;
                dpsi += t * inverse;
            } else {
                phi += t;
                dphi += t * inverse;
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

int cut_block(Tree& tree, std::vector<double>& d, const std::vector<double>& e, int leaf, int offset, int size) {
    if (size <= leaf) {
        tree.leaves.push_back(offset);
        tree.leaves.push_back(size);
        return 0;
    }
    const int half = size / 2, middle = offset + half;
    const double coupling = e[static_cast<std::size_t>(middle - 1)], rho = std::abs(coupling);
    d[static_cast<std::size_t>(middle - 1)] -= rho;
    d[static_cast<std::size_t>(middle)] -= rho;
    const int height = 1 + std::max(cut_block(tree, d, e, leaf, offset, half), cut_block(tree, d, e, leaf, middle, size - half));
    tree.merges.push_back({offset, size, half, height, rho, coupling < 0.0 ? -1.0 : 1.0});
    tree.height = std::max(tree.height, height);
    return height;
}

}  // namespace

Tree cut(std::vector<double>& d, const std::vector<double>& e, int leaf) {
    Tree tree;
    if (d.empty()) return tree;
    cut_block(tree, d, e, std::max(leaf, 2), 0, static_cast<int>(d.size()));
    std::stable_sort(tree.merges.begin(), tree.merges.end(),
                     [](const Merge& a, const Merge& b) { return a.height < b.height; });
    return tree;
}

Deflation deflate(double* d, int size, std::vector<double>& z, double rho0) {
    Deflation f;
    const std::size_t n = static_cast<std::size_t>(size);
    // z to unit length: its two rows are unit vectors
    double length = 0.0;
    for (const double x : z) length += x * x;
    length = std::sqrt(length);
    for (double& x : z) x /= length;
    f.rho = rho0 * length * length;
    // the poles ascending
    std::vector<int> by(n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return d[a] < d[b]; });
    double dmax = 0.0, zmax = 0.0;
    for (int j = 0; j < size; ++j) {
        dmax = std::max(dmax, std::abs(d[j]));
        zmax = std::max(zmax, std::abs(z[static_cast<std::size_t>(j)]));
    }
    const double tolerance = 8.0 * kEps * std::max(dmax, zmax);
    f.values.assign(n, 0.0);
    if (f.rho * zmax <= tolerance) {
        // nothing couples the halves
        for (const int j : by) f.deflated.push_back(j);
        for (int j = 0; j < size; ++j) f.values[static_cast<std::size_t>(j)] = d[j];
    } else {
        int candidate = -1;
        for (const int j : by) {
            if (f.rho * std::abs(z[static_cast<std::size_t>(j)]) <= tolerance) {
                f.deflated.push_back(j);
                f.values[static_cast<std::size_t>(j)] = d[j];
                continue;
            }
            if (candidate < 0) {
                candidate = j;
                continue;
            }
            // two poles within the tolerance of each other: a rotation leaves
            // the first without a component
            const double s0 = z[static_cast<std::size_t>(candidate)], c0 = z[static_cast<std::size_t>(j)];
            const double tau = std::hypot(c0, s0), gap = d[j] - d[candidate];
            const double c = c0 / tau, s = -s0 / tau;
            if (std::abs(gap * c * s) <= tolerance) {
                z[static_cast<std::size_t>(j)] = tau;
                z[static_cast<std::size_t>(candidate)] = 0.0;
                f.rotations.push_back({candidate, j, c, s});
                const double first = d[candidate] * c * c + d[j] * s * s;
                d[j] = d[candidate] * s * s + d[j] * c * c;
                d[candidate] = first;
                f.deflated.push_back(candidate);
                f.values[static_cast<std::size_t>(candidate)] = first;
            } else {
                f.kept.push_back(candidate);
            }
            candidate = j;
        }
        if (candidate >= 0) f.kept.push_back(candidate);
    }
    for (const int j : f.kept) {
        f.pole.push_back(d[j]);
        f.w.push_back(z[static_cast<std::size_t>(j)]);
    }
    return f;
}

void secular(const Deflation& f, double* roots, double* u) {
    const int k = static_cast<int>(f.kept.size());
    if (k == 0) return;
    const std::size_t count = static_cast<std::size_t>(k);
    const double* pole = f.pole.data();
    std::vector<double> w2(count);
    for (std::size_t j = 0; j < count; ++j) w2[j] = f.w[j] * f.w[j];
    // column i of u holds d_j - root_i
    for (int i = 0; i < k; ++i) roots[i] = secular_root(i, k, pole, w2.data(), f.rho, u + static_cast<std::size_t>(i) * count);
    // w from the roots: w_j^2 = prod_i (root_i - d_j) / (rho prod_(i != j) (d_i - d_j))
    std::vector<double> fitted(count);
    for (int j = 0; j < k; ++j) {
        double product = -u[static_cast<std::size_t>(j) * count + j] / f.rho;
        for (int i = 0; i < k; ++i)
            if (i != j) product *= u[static_cast<std::size_t>(i) * count + j] / (pole[j] - pole[i]);
        fitted[static_cast<std::size_t>(j)] = std::copysign(std::sqrt(std::abs(product)), f.w[static_cast<std::size_t>(j)]);
    }
    for (int i = 0; i < k; ++i) {
        double* column = u + static_cast<std::size_t>(i) * count;
        double norm = 0.0;
        for (int j = 0; j < k; ++j) {
            column[j] = fitted[static_cast<std::size_t>(j)] / column[j];
            norm += column[j] * column[j];
        }
        norm = 1.0 / std::sqrt(norm);
        for (int j = 0; j < k; ++j) column[j] *= norm;
    }
}

}  // namespace mdapy::dft::host

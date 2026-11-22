// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/mixer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// x of A x = b, A symmetric positive semidefinite m x m: by its eigenvectors
// (Jacobi), directions of eigenvalues below 1e-12 of the largest dropped
std::vector<double> solve_symmetric(std::vector<double> a, const std::vector<double>& b, int m) {
    std::vector<double> v(static_cast<std::size_t>(m * m), 0.0);
    for (int i = 0; i < m; ++i) v[static_cast<std::size_t>(i * m + i)] = 1.0;
    auto at = [&](std::vector<double>& x, int i, int j) -> double& { return x[static_cast<std::size_t>(i * m + j)]; };
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j) off += at(a, i, j) * at(a, i, j);
        if (off < 1e-30) break;
        for (int p = 0; p < m; ++p)
            for (int q = p + 1; q < m; ++q) {
                if (std::abs(at(a, p, q)) < 1e-300) continue;
                const double theta = (at(a, q, q) - at(a, p, p)) / (2.0 * at(a, p, q));
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < m; ++k) {
                    const double akp = at(a, k, p), akq = at(a, k, q);
                    at(a, k, p) = c * akp - s * akq;
                    at(a, k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < m; ++k) {
                    const double apk = at(a, p, k), aqk = at(a, q, k);
                    at(a, p, k) = c * apk - s * aqk;
                    at(a, q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < m; ++k) {
                    const double vkp = at(v, k, p), vkq = at(v, k, q);
                    at(v, k, p) = c * vkp - s * vkq;
                    at(v, k, q) = s * vkp + c * vkq;
                }
            }
    }
    double largest = 0.0;
    for (int i = 0; i < m; ++i) largest = std::max(largest, at(a, i, i));
    std::vector<double> x(static_cast<std::size_t>(m), 0.0);
    for (int e = 0; e < m; ++e) {
        const double lambda = at(a, e, e);
        if (!(lambda > 1e-12 * largest)) continue;
        double proj = 0.0;
        for (int k = 0; k < m; ++k) proj += at(v, k, e) * b[static_cast<std::size_t>(k)];
        for (int k = 0; k < m; ++k) x[static_cast<std::size_t>(k)] += at(v, k, e) * proj / lambda;
    }
    return x;
}

}  // namespace

Mixer::Mixer(std::vector<double> g2, double volume, int channels, MixingOptions options)
    : g2_(std::move(g2)), volume_(volume), channels_(channels), options_(options), n_(g2_.size()) {
    if (channels < 1 || channels > 2) throw std::invalid_argument("one or two channels of density");
    if (n_ < 2 || g2_[0] != 0.0) throw std::invalid_argument("the sphere of the density starts at G = 0");
    if (!(options_.beta > 0.0) || options_.history < 1 || !(options_.metric_weight >= 1.0) ||
        options_.kerker < 0.0 || options_.minimum < 0.0 || !(options_.magnetic_beta > 0.0) ||
        options_.magnetic_kerker < 0.0)
        throw std::invalid_argument("mixing: beta and magnetic_beta > 0, history >= 1, metric_weight >= 1, "
                                    "kerker >= 0, minimum >= 0");
    double shortest = 0.0;
    for (std::size_t g = 1; g < n_; ++g)
        if (g2_[g] > 0.0 && (shortest == 0.0 || g2_[g] < shortest)) shortest = g2_[g];
    const double q1 = (options_.metric_weight - 1.0) * shortest;
    metric_.assign(n_ * channels_, 1.0);
    precondition_.assign(n_ * channels_, options_.magnetic_beta);
    if (channels_ == 2 && options_.magnetic_kerker > 0.0)
        for (std::size_t g = 1; g < n_; ++g) {
            const double x = g2_[g], q = options_.magnetic_kerker;
            precondition_[n_ + g] = options_.magnetic_beta * x / (x + q * q);
        }
    for (std::size_t g = 0; g < n_; ++g) {
        const double x = g2_[g];
        metric_[g] = g == 0 ? 0.0 : (x + q1) / x;
        precondition_[g] = g == 0 ? 0.0
                                  : (options_.kerker > 0.0
                                         ? std::max(options_.minimum,
                                                    options_.beta * x / (x + options_.kerker * options_.kerker))
                                         : options_.beta);
    }
}

void Mixer::reset() {
    inputs_.clear();
    residuals_.clear();
}

double Mixer::dot(const std::vector<cuDoubleComplex>& a, const std::vector<cuDoubleComplex>& b) const {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += metric_[i] * (a[i].x * b[i].x + a[i].y * b[i].y);
    return s;
}

double Mixer::residual_energy(const std::vector<cuDoubleComplex>& in,
                              const std::vector<cuDoubleComplex>& out) const {
    double s = 0.0;
    for (std::size_t g = 1; g < n_; ++g) {
        const double re = out[g].x - in[g].x, im = out[g].y - in[g].y;
        s += (re * re + im * im) / g2_[g];
    }
    if (channels_ == 2)
        for (std::size_t g = n_; g < 2 * n_; ++g) {
            const double re = out[g].x - in[g].x, im = out[g].y - in[g].y;
            s += re * re + im * im;
        }
    return 2.0 * kPi * volume_ * s;
}

std::vector<cuDoubleComplex> Mixer::mix(const std::vector<cuDoubleComplex>& in,
                                        const std::vector<cuDoubleComplex>& out) {
    const std::size_t size = n_ * channels_;
    if (in.size() != size || out.size() != size) throw std::invalid_argument("densities of the sphere");
    std::vector<cuDoubleComplex> r(size);
    for (std::size_t i = 0; i < size; ++i) r[i] = make_cuDoubleComplex(out[i].x - in[i].x, out[i].y - in[i].y);
    inputs_.push_back(in);
    residuals_.push_back(r);
    while (static_cast<int>(inputs_.size()) > options_.history + 1) {
        inputs_.pop_front();
        residuals_.pop_front();
    }
    const int m = static_cast<int>(inputs_.size()) - 1;  // differences
    std::vector<cuDoubleComplex> rho = in, res = r;
    if (m > 0) {
        std::vector<std::vector<cuDoubleComplex>> dr(static_cast<std::size_t>(m)), drho(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i) {
            auto& a = dr[static_cast<std::size_t>(i)];
            auto& b = drho[static_cast<std::size_t>(i)];
            a.resize(size);
            b.resize(size);
            const auto& r0 = residuals_[static_cast<std::size_t>(i)];
            const auto& r1 = residuals_[static_cast<std::size_t>(i + 1)];
            const auto& p0 = inputs_[static_cast<std::size_t>(i)];
            const auto& p1 = inputs_[static_cast<std::size_t>(i + 1)];
            for (std::size_t j = 0; j < size; ++j) {
                a[j] = make_cuDoubleComplex(r1[j].x - r0[j].x, r1[j].y - r0[j].y);
                b[j] = make_cuDoubleComplex(p1[j].x - p0[j].x, p1[j].y - p0[j].y);
            }
        }
        std::vector<double> a(static_cast<std::size_t>(m * m)), b(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i) {
            b[static_cast<std::size_t>(i)] = dot(dr[static_cast<std::size_t>(i)], r);
            for (int j = 0; j <= i; ++j)
                a[static_cast<std::size_t>(i * m + j)] = a[static_cast<std::size_t>(j * m + i)] =
                    dot(dr[static_cast<std::size_t>(i)], dr[static_cast<std::size_t>(j)]);
        }
        const std::vector<double> gamma = solve_symmetric(a, b, m);
        for (int i = 0; i < m; ++i) {
            const double c = gamma[static_cast<std::size_t>(i)];
            for (std::size_t j = 0; j < size; ++j) {
                rho[j].x -= c * drho[static_cast<std::size_t>(i)][j].x;
                rho[j].y -= c * drho[static_cast<std::size_t>(i)][j].y;
                res[j].x -= c * dr[static_cast<std::size_t>(i)][j].x;
                res[j].y -= c * dr[static_cast<std::size_t>(i)][j].y;
            }
        }
    }
    for (std::size_t j = 0; j < size; ++j) {
        rho[j].x += precondition_[j] * res[j].x;
        rho[j].y += precondition_[j] * res[j].y;
    }
    rho[0] = in[0];
    return rho;
}

}  // namespace mdapy::dft

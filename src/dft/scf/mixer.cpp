// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/mixer.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// x of A x = b, A symmetric positive semidefinite m x m: by its eigenvectors
// (Jacobi), directions of eigenvalues below cut times the largest dropped
std::vector<double> solve_symmetric(std::vector<double> a, const std::vector<double>& b, int m, double cut) {
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
        if (!(lambda > cut * largest)) continue;
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
        options_.magnetic_kerker < 0.0 || !(options_.conditioning >= 1.0))
        throw std::invalid_argument("mixing: beta and magnetic_beta > 0, history >= 1, metric_weight >= 1, "
                                    "kerker >= 0, minimum >= 0, conditioning >= 1");
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
    extra_inputs_.clear();
    extra_residuals_.clear();
}

void Mixer::set_preconditioner(std::function<void(std::vector<cuDoubleComplex>&)> apply) {
    preconditioner_ = std::move(apply);
}

void Mixer::set_magnetic_coupling(
    std::function<void(const std::vector<cuDoubleComplex>&, std::vector<cuDoubleComplex>&)> couple) {
    coupling_ = std::move(couple);
}

std::vector<cuDoubleComplex> gmres(
    const std::function<std::vector<cuDoubleComplex>(const std::vector<cuDoubleComplex>&)>& apply,
    const std::vector<cuDoubleComplex>& b, int iterations, double tolerance) {
    using C = std::complex<double>;
    const std::size_t n = b.size();
    auto norm = [](const std::vector<C>& v) {
        double s = 0.0;
        for (const auto& x : v) s += std::norm(x);
        return std::sqrt(s);
    };
    std::vector<C> r(n);
    for (std::size_t i = 0; i < n; ++i) r[i] = C(b[i].x, b[i].y);
    const double beta = norm(r);
    std::vector<cuDoubleComplex> x(n, make_cuDoubleComplex(0.0, 0.0));
    if (!(beta > 0.0) || iterations < 1) return x;
    const int m = iterations;
    std::vector<std::vector<C>> v;
    v.reserve(static_cast<std::size_t>(m) + 1);
    for (auto& e : r) e /= beta;
    v.push_back(std::move(r));
    // the Hessenberg matrix by columns, its Givens rotations and the residual
    std::vector<std::vector<C>> h(static_cast<std::size_t>(m), std::vector<C>(static_cast<std::size_t>(m) + 1));
    std::vector<double> cs(static_cast<std::size_t>(m));
    std::vector<C> sn(static_cast<std::size_t>(m)), g(static_cast<std::size_t>(m) + 1);
    g[0] = beta;
    int k = 0;
    std::vector<cuDoubleComplex> in(n);
    for (int j = 0; j < m; ++j) {
        const auto& vj = v[static_cast<std::size_t>(j)];
        for (std::size_t i = 0; i < n; ++i) in[i] = make_cuDoubleComplex(vj[i].real(), vj[i].imag());
        const auto out = apply(in);
        std::vector<C> w(n);
        for (std::size_t i = 0; i < n; ++i) w[i] = C(out[i].x, out[i].y);
        auto& col = h[static_cast<std::size_t>(j)];
        for (int i = 0; i <= j; ++i) {  // modified Gram-Schmidt
            const auto& vi = v[static_cast<std::size_t>(i)];
            C d(0.0, 0.0);
            for (std::size_t q = 0; q < n; ++q) d += std::conj(vi[q]) * w[q];
            col[static_cast<std::size_t>(i)] = d;
            for (std::size_t q = 0; q < n; ++q) w[q] -= d * vi[q];
        }
        const double next = norm(w);
        col[static_cast<std::size_t>(j) + 1] = next;
        for (int i = 0; i < j; ++i) {  // the rotations so far
            const C a = col[static_cast<std::size_t>(i)], c2 = col[static_cast<std::size_t>(i) + 1];
            col[static_cast<std::size_t>(i)] = cs[static_cast<std::size_t>(i)] * a + sn[static_cast<std::size_t>(i)] * c2;
            col[static_cast<std::size_t>(i) + 1] =
                -std::conj(sn[static_cast<std::size_t>(i)]) * a + cs[static_cast<std::size_t>(i)] * c2;
        }
        const C a = col[static_cast<std::size_t>(j)], bb = col[static_cast<std::size_t>(j) + 1];
        const double rr = std::sqrt(std::norm(a) + std::norm(bb));
        if (std::abs(a) == 0.0) {
            cs[static_cast<std::size_t>(j)] = 0.0;
            sn[static_cast<std::size_t>(j)] = 1.0;
        } else {
            cs[static_cast<std::size_t>(j)] = std::abs(a) / rr;
            sn[static_cast<std::size_t>(j)] = a * std::conj(bb) / (rr * std::abs(a));
        }
        col[static_cast<std::size_t>(j)] = cs[static_cast<std::size_t>(j)] * a + sn[static_cast<std::size_t>(j)] * bb;
        col[static_cast<std::size_t>(j) + 1] = 0.0;
        g[static_cast<std::size_t>(j) + 1] = -std::conj(sn[static_cast<std::size_t>(j)]) * g[static_cast<std::size_t>(j)];
        g[static_cast<std::size_t>(j)] *= cs[static_cast<std::size_t>(j)];
        k = j + 1;
        if (std::abs(g[static_cast<std::size_t>(j) + 1]) < tolerance * beta || !(next > 0.0)) break;
        for (auto& e : w) e /= next;
        v.push_back(std::move(w));
    }
    // H y = g, upper triangular k x k, and x = V y
    std::vector<C> y(static_cast<std::size_t>(k));
    for (int i = k - 1; i >= 0; --i) {
        C s = g[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < k; ++j) s -= h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(j)];
        y[static_cast<std::size_t>(i)] = s / h[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < k; ++i)
        for (std::size_t q = 0; q < n; ++q) {
            const C t = y[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(i)][q];
            x[q].x += t.real();
            x[q].y += t.imag();
        }
    return x;
}

void Mixer::set_scale(double factor) {
    if (!(factor > 0.0)) throw std::invalid_argument("the scale of the mixing must be positive");
    scale_ = factor;
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
                                        const std::vector<cuDoubleComplex>& out, std::vector<double>* extra,
                                        const std::vector<double>* extra_out) {
    const std::size_t size = n_ * channels_;
    if (in.size() != size || out.size() != size) throw std::invalid_argument("densities of the sphere");
    const bool along = extra != nullptr;
    if (along != (extra_out != nullptr) || (along && (extra->size() != extra_out->size() ||
                                                      extra->size() % static_cast<std::size_t>(channels_) != 0)))
        throw std::invalid_argument("the components that come along: their input and output, channels of one size");
    if (along != !extra_inputs_.empty() && !inputs_.empty())
        throw std::invalid_argument("components that come along in every step or none");
    std::vector<cuDoubleComplex> r(size);
    for (std::size_t i = 0; i < size; ++i) r[i] = make_cuDoubleComplex(out[i].x - in[i].x, out[i].y - in[i].y);
    inputs_.push_back(in);
    residuals_.push_back(r);
    if (along) {
        std::vector<double> er(extra->size());
        for (std::size_t i = 0; i < er.size(); ++i) er[i] = (*extra_out)[i] - (*extra)[i];
        extra_inputs_.push_back(*extra);
        extra_residuals_.push_back(std::move(er));
    }
    while (static_cast<int>(inputs_.size()) > options_.history + 1) {
        inputs_.pop_front();
        residuals_.pop_front();
        if (along) {
            extra_inputs_.pop_front();
            extra_residuals_.pop_front();
        }
    }
    std::vector<double> x, xr;
    if (along) {
        x = extra_inputs_.back();
        xr = extra_residuals_.back();
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
        // A = dR^T dR: its eigenvalues are the squares of the singular values
        const std::vector<double> gamma =
            solve_symmetric(a, b, m, 1.0 / (options_.conditioning * options_.conditioning));
        for (int i = 0; i < m; ++i) {
            const double c = gamma[static_cast<std::size_t>(i)];
            if (along) {
                const auto& x0 = extra_inputs_[static_cast<std::size_t>(i)];
                const auto& x1 = extra_inputs_[static_cast<std::size_t>(i + 1)];
                const auto& r0 = extra_residuals_[static_cast<std::size_t>(i)];
                const auto& r1 = extra_residuals_[static_cast<std::size_t>(i + 1)];
                for (std::size_t j = 0; j < x.size(); ++j) {
                    x[j] -= c * (x1[j] - x0[j]);
                    xr[j] -= c * (r1[j] - r0[j]);
                }
            }
            for (std::size_t j = 0; j < size; ++j) {
                rho[j].x -= c * drho[static_cast<std::size_t>(i)][j].x;
                rho[j].y -= c * drho[static_cast<std::size_t>(i)][j].y;
                res[j].x -= c * dr[static_cast<std::size_t>(i)][j].x;
                res[j].y -= c * dr[static_cast<std::size_t>(i)][j].y;
            }
        }
    }
    std::size_t from = 0;
    if (preconditioner_) {  // the total density by the preconditioner given
        std::vector<cuDoubleComplex> total(res.begin(), res.begin() + static_cast<std::ptrdiff_t>(n_));
        total[0] = make_cuDoubleComplex(0.0, 0.0);
        preconditioner_(total);
        for (std::size_t j = 0; j < n_; ++j) {
            rho[j].x += scale_ * options_.beta * total[j].x;
            rho[j].y += scale_ * options_.beta * total[j].y;
        }
        from = n_;
        if (coupling_ && channels_ == 2) {
            std::vector<cuDoubleComplex> dm(res.begin() + static_cast<std::ptrdiff_t>(n_),
                                            res.begin() + static_cast<std::ptrdiff_t>(2 * n_));
            coupling_(total, dm);
            for (std::size_t j = 0; j < n_; ++j) {
                rho[n_ + j].x += scale_ * precondition_[n_ + j] * dm[j].x;
                rho[n_ + j].y += scale_ * precondition_[n_ + j] * dm[j].y;
            }
            from = 2 * n_;
        }
    }
    for (std::size_t j = from; j < size; ++j) {
        rho[j].x += scale_ * precondition_[j] * res[j].x;
        rho[j].y += scale_ * precondition_[j] * res[j].y;
    }
    rho[0] = in[0];
    if (along) {
        const std::size_t per = x.size() / static_cast<std::size_t>(channels_);
        for (std::size_t j = 0; j < x.size(); ++j)
            (*extra)[j] = x[j] + scale_ * (j < per ? options_.beta : options_.magnetic_beta) * xr[j];
    }
    return rho;
}

}  // namespace mdapy::dft

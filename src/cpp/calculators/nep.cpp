// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "calculators/nep.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mdapy::calculators {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Four-body invariant of the l = 2 harmonics, the Wigner 3j coupling
// (2 2 2; m1 m2 m3) written for the real harmonics s0 (m = 0), s1, s2
// (m = 1, cosine and sine), s3, s4 (m = 2):
// alpha (-2 s0^3 - 3 s0 (s1^2 + s2^2) + 6 s0 (s3^2 + s4^2))
//   + beta (s3 (s2^2 - s1^2) - 2 s1 s2 s4).
const double kAlpha = 1.0 / std::sqrt(70.0);
const double kBeta = 3.0 * std::sqrt(210.0) / 70.0;
// Five-body invariant of the l = 1 harmonics, the couplings through
// l = 0, 1, 2 summed: (7 / 15) (s0^2 + s1^2 + s2^2)^2.
constexpr double kFive = 7.0 / 15.0;

// f_k(r) = (T_k(x) + 1) fc(r) / 2 with x = 2 (r / rc - 1)^2 - 1 and the
// cosine cutoff fc, and the derivatives by r, for k = 0..k_max.
void chebyshev_basis(double r, double rc, int k_max, double* f, double* fp) {
    const double u = r / rc - 1.0;
    const double x = 2.0 * u * u - 1.0;
    const double dx = 4.0 * u / rc;
    const double fc = 0.5 * std::cos(kPi * r / rc) + 0.5;
    const double fcp = -0.5 * kPi / rc * std::sin(kPi * r / rc);
    f[0] = fc;
    fp[0] = fcp;
    double t0 = 1.0, t1 = x;  // T_{k-2} and T_{k-1}
    double d0 = 0.0, d1 = 1.0;  // and their derivatives by x
    for (int k = 1; k <= k_max; ++k) {
        if (k >= 2) {
            const double t2 = 2.0 * x * t1 - t0;
            const double d2 = 2.0 * t1 + 2.0 * x * d1 - d0;
            t0 = t1;
            t1 = t2;
            d0 = d1;
            d1 = d2;
        }
        f[k] = 0.5 * (t1 + 1.0) * fc;
        fp[k] = 0.5 * (d1 * dx * fc + (t1 + 1.0) * fcp);
    }
}

// g_n = sum_k c[n][k] f_k for n < count.
void contract(const double* c, int count, int k_count, const double* f, double* g) {
    for (int n = 0; n < count; ++n) {
        double sum = 0.0;
        for (int k = 0; k < k_count; ++k) {
            sum += c[n * k_count + k] * f[k];
        }
        g[n] = sum;
    }
}

// The real spherical harmonics of the unit vector u, in the order of
// NepModel::harmonics with the cosine before the sine of each m > 0, and,
// unless gradient is null, their gradients (h x 3) by the vector u * r.
void evaluate_harmonics(const NepModel& model, const double* u, double r, double* y,
                        double* gradient) {
    const int l_max = model.l_max;
    double re[9], im[9], zp[9];
    re[0] = 1.0;
    im[0] = 0.0;
    zp[0] = 1.0;
    for (int m = 1; m <= l_max; ++m) {
        re[m] = re[m - 1] * u[0] - im[m - 1] * u[1];
        im[m] = re[m - 1] * u[1] + im[m - 1] * u[0];
        zp[m] = zp[m - 1] * u[2];
    }
    const double* c = model.harmonics.data();
    int h = 0;
    for (int l = 1; l <= l_max; ++l) {
        for (int m = 0; m <= l; ++m) {
            double p = 0.0, dp = 0.0;
            for (int k = 0; k <= l - m; ++k) {
                p += c[k] * zp[k];
                if (k > 0) {
                    dp += k * c[k] * zp[k - 1];
                }
            }
            c += l - m + 1;
            double g[2][3];
            if (m == 0) {
                y[h] = p;
                g[0][0] = 0.0;
                g[0][1] = 0.0;
                g[0][2] = dp;
            } else {
                y[h] = p * re[m];
                y[h + 1] = p * im[m];
                g[0][0] = p * m * re[m - 1];
                g[0][1] = -p * m * im[m - 1];
                g[0][2] = dp * re[m];
                g[1][0] = p * m * im[m - 1];
                g[1][1] = p * m * re[m - 1];
                g[1][2] = dp * im[m];
            }
            const int parts = m == 0 ? 1 : 2;
            if (gradient != nullptr) {
                // the gradient of y(r / |r|) is the tangential part over |r|
                for (int s = 0; s < parts; ++s) {
                    const double radial = g[s][0] * u[0] + g[s][1] * u[1] + g[s][2] * u[2];
                    for (int a = 0; a < 3; ++a) {
                        gradient[3 * (h + s) + a] = (g[s][a] - radial * u[a]) / r;
                    }
                }
            }
            h += parts;
        }
    }
}

// The ZBL energy of a pair and its derivative by r; row from NepModel::zbl.
void zbl_pair(const double* row, double r, double& energy, double& derivative) {
    const double inner = row[0], outer = row[1];
    double s = 1.0, sp = 0.0;
    if (r >= outer) {
        energy = derivative = 0.0;
        return;
    }
    if (r >= inner) {
        const double w = kPi / (outer - inner);
        s = 0.5 * std::cos(w * (r - inner)) + 0.5;
        sp = -0.5 * w * std::sin(w * (r - inner));
    }
    const double x = r * row[3];
    double phi = 0.0, phip = 0.0;
    for (int k = 0; k < 4; ++k) {
        const double term = row[4 + 2 * k] * std::exp(-row[5 + 2 * k] * x);
        phi += term;
        phip -= row[5 + 2 * k] * term;
    }
    const double v = row[2] * phi / r;
    const double vp = row[2] * (phip * row[3] / r - phi / (r * r));
    energy = v * s;
    derivative = vp * s + v * sp;
}

}  // namespace

void NepModel::prepare() {
    if (types < 1 || l_max < 1 || l_max > 8 || neurons < 1 || n_radial < 0 || n_angular < 0 ||
        k_radial < 0 || k_angular < 0 || (four_body && l_max < 2) ||
        (five_body && !four_body)) {
        throw std::invalid_argument("invalid NEP sizes");
    }
    const auto t = static_cast<std::size_t>(types);
    dim = n_radial + 1 + (n_angular + 1) * (l_max + four_body + five_body);
    if (rc_radial.size() != t || rc_angular.size() != t ||
        w0.size() != t * neurons * dim || b0.size() != t * neurons ||
        w1.size() != t * neurons ||
        c_radial.size() != t * t * (n_radial + 1) * (k_radial + 1) ||
        c_angular.size() != t * t * (n_angular + 1) * (k_angular + 1) ||
        q_scaler.size() != static_cast<std::size_t>(dim) ||
        (!zbl.empty() && zbl.size() != t * t * 12)) {
        throw std::invalid_argument("inconsistent NEP parameters");
    }
    // Legendre polynomials by their recurrence, then the m-th derivatives
    // scaled to orthonormal real harmonics.
    std::vector<std::vector<double>> legendre{{1.0}, {0.0, 1.0}};
    for (int l = 1; l < l_max; ++l) {
        std::vector<double> next(static_cast<std::size_t>(l) + 2, 0.0);
        for (int k = 0; k <= l + 1; ++k) {
            double value = 0.0;
            if (k >= 1) {
                value += (2 * l + 1) * legendre[l][k - 1];
            }
            if (k <= l - 1) {
                value -= l * legendre[l - 1][k];
            }
            next[k] = value / (l + 1);
        }
        legendre.push_back(next);
    }
    harmonics.clear();
    for (int l = 1; l <= l_max; ++l) {
        std::vector<double> p = legendre[l];
        double ratio = 1.0;  // (l - m)! / (l + m)!
        for (int m = 0; m <= l; ++m) {
            if (m > 0) {
                for (std::size_t k = 0; k + 1 < p.size(); ++k) {
                    p[k] = (k + 1) * p[k + 1];
                }
                p.pop_back();
                ratio /= static_cast<double>((l + m) * (l - m + 1));
            }
            const double norm =
                std::sqrt((m == 0 ? 1.0 : 2.0) * (2 * l + 1) / (4.0 * kPi) * ratio);
            for (double coefficient : p) {
                harmonics.push_back(norm * coefficient);
            }
        }
    }
}

void compute_nep(const NepModel& model, std::int64_t n, const std::int32_t* species,
                 const std::int64_t* offsets, const std::int32_t* neighbors,
                 const double* vectors, double* energies, double* forces, double* virials,
                 double* descriptors, const Control& control) {
    const int types = model.types, dim = model.dim, harmonics = model.harmonic_count();
    const int nr = model.n_radial + 1, na = model.n_angular + 1;
    const int kr = model.k_radial + 1, ka = model.k_angular + 1;
    const int l_max = model.l_max, neurons = model.neurons;
    const int k_size = std::max(kr, ka), n_size = std::max(nr, na);
    const bool zbl = !model.zbl.empty();
    std::vector<double> parity(static_cast<std::size_t>(harmonics));
    for (int l = 1; l <= l_max; ++l) {
        for (int h = l * l - 1; h < (l + 1) * (l + 1) - 1; ++h) {
            parity[h] = l % 2 == 0 ? 1.0 : -1.0;
        }
    }
    // dE_i / dq_n of the radial descriptors and dE_i / ds_nh of the sums of
    // harmonics, kept for the forces on the neighbors.
    std::vector<double> radial_slope(static_cast<std::size_t>(n) * nr);
    std::vector<double> angular_slope(static_cast<std::size_t>(n) * na * harmonics);

    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> f(k_size), fp(k_size), g(n_size), y(harmonics);
            std::vector<double> s(static_cast<std::size_t>(na) * harmonics), q(dim), slope(dim);
            for (std::int64_t i = begin; i < end; ++i) {
                const int ti = species[i];
                std::fill(q.begin(), q.end(), 0.0);
                std::fill(s.begin(), s.end(), 0.0);
                double repulsion = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const int tj = species[neighbors[p]];
                    const double* v = vectors + 3 * p;
                    const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    const double rc_r = 0.5 * (model.rc_radial[ti] + model.rc_radial[tj]);
                    if (r < rc_r) {
                        chebyshev_basis(r, rc_r, kr - 1, f.data(), fp.data());
                        contract(&model.c_radial[static_cast<std::size_t>(ti * types + tj) * nr * kr],
                                 nr, kr, f.data(), g.data());
                        for (int k = 0; k < nr; ++k) {
                            q[k] += g[k];
                        }
                    }
                    const double rc_a = 0.5 * (model.rc_angular[ti] + model.rc_angular[tj]);
                    if (r < rc_a) {
                        chebyshev_basis(r, rc_a, ka - 1, f.data(), fp.data());
                        contract(&model.c_angular[static_cast<std::size_t>(ti * types + tj) * na * ka],
                                 na, ka, f.data(), g.data());
                        const double u[3] = {v[0] / r, v[1] / r, v[2] / r};
                        evaluate_harmonics(model, u, r, y.data(), nullptr);
                        for (int k = 0; k < na; ++k) {
                            for (int h = 0; h < harmonics; ++h) {
                                s[k * harmonics + h] += g[k] * y[h];
                            }
                        }
                        if (zbl) {
                            double e, de;
                            zbl_pair(&model.zbl[static_cast<std::size_t>(ti * types + tj) * 12], r,
                                     e, de);
                            repulsion += 0.5 * e;
                        }
                    }
                }
                // the invariants: per l, then four-body, then five-body
                for (int l = 1; l <= l_max; ++l) {
                    for (int k = 0; k < na; ++k) {
                        const double* sk = &s[k * harmonics];
                        double sum = 0.0;
                        for (int h = l * l - 1; h < (l + 1) * (l + 1) - 1; ++h) {
                            sum += sk[h] * sk[h];
                        }
                        q[nr + (l - 1) * na + k] = sum;
                    }
                }
                for (int k = 0; k < na; ++k) {
                    const double* sk = &s[k * harmonics];
                    if (model.four_body) {
                        const double* a = sk + 3;
                        q[nr + l_max * na + k] =
                            kAlpha * a[0] *
                                (-2.0 * a[0] * a[0] - 3.0 * (a[1] * a[1] + a[2] * a[2]) +
                                 6.0 * (a[3] * a[3] + a[4] * a[4])) +
                            kBeta * (a[3] * (a[2] * a[2] - a[1] * a[1]) - 2.0 * a[1] * a[2] * a[4]);
                    }
                    if (model.five_body) {
                        const double sum = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];
                        q[nr + (l_max + 1) * na + k] = kFive * sum * sum;
                    }
                }
                for (int d = 0; d < dim; ++d) {
                    q[d] *= model.q_scaler[d];
                }
                if (descriptors != nullptr) {
                    std::copy(q.begin(), q.end(), descriptors + i * dim);
                }

                // one hidden layer of tanh neurons
                double energy = -model.b1;
                std::fill(slope.begin(), slope.end(), 0.0);
                for (int k = 0; k < neurons; ++k) {
                    const std::size_t at = static_cast<std::size_t>(ti) * neurons + k;
                    const double* w = &model.w0[at * dim];
                    double x = -model.b0[at];
                    for (int d = 0; d < dim; ++d) {
                        x += w[d] * q[d];
                    }
                    const double t = std::tanh(x);
                    energy += model.w1[at] * t;
                    const double factor = model.w1[at] * (1.0 - t * t);
                    for (int d = 0; d < dim; ++d) {
                        slope[d] += factor * w[d];
                    }
                }
                for (int d = 0; d < dim; ++d) {
                    slope[d] *= model.q_scaler[d];
                }
                energies[i] = energy + repulsion;

                std::copy(slope.begin(), slope.begin() + nr, &radial_slope[i * nr]);
                double* gi = &angular_slope[static_cast<std::size_t>(i) * na * harmonics];
                for (int k = 0; k < na; ++k) {
                    const double* sk = &s[k * harmonics];
                    double* gk = gi + k * harmonics;
                    for (int l = 1; l <= l_max; ++l) {
                        const double e = slope[nr + (l - 1) * na + k];
                        for (int h = l * l - 1; h < (l + 1) * (l + 1) - 1; ++h) {
                            gk[h] = 2.0 * e * sk[h];
                        }
                    }
                    if (model.four_body) {
                        const double e = slope[nr + l_max * na + k];
                        const double* a = sk + 3;
                        double* ga = gk + 3;
                        ga[0] += e * kAlpha *
                                 (-6.0 * a[0] * a[0] - 3.0 * (a[1] * a[1] + a[2] * a[2]) +
                                  6.0 * (a[3] * a[3] + a[4] * a[4]));
                        ga[1] += e * (-6.0 * kAlpha * a[0] * a[1] -
                                      2.0 * kBeta * (a[3] * a[1] + a[2] * a[4]));
                        ga[2] += e * (-6.0 * kAlpha * a[0] * a[2] +
                                      2.0 * kBeta * (a[3] * a[2] - a[1] * a[4]));
                        ga[3] += e * (12.0 * kAlpha * a[0] * a[3] +
                                      kBeta * (a[2] * a[2] - a[1] * a[1]));
                        ga[4] += e * (12.0 * kAlpha * a[0] * a[4] - 2.0 * kBeta * a[1] * a[2]);
                    }
                    if (model.five_body) {
                        const double e = slope[nr + (l_max + 1) * na + k];
                        const double sum = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];
                        for (int h = 0; h < 3; ++h) {
                            gk[h] += e * 4.0 * kFive * sum * sk[h];
                        }
                    }
                }
            }
        },
        control);

    // Forces and virials; each atom gathers the derivatives of its own
    // energy and of its neighbors' energies by their pair vectors.
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> f(k_size), fp(k_size), g_ij(n_size), gp_ij(n_size), g_ji(n_size),
                gp_ji(n_size), y(harmonics), gradient(3 * static_cast<std::size_t>(harmonics));
            for (std::int64_t i = begin; i < end; ++i) {
                const int ti = species[i];
                double force[3] = {0.0, 0.0, 0.0};
                double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                const double* gi = &angular_slope[static_cast<std::size_t>(i) * na * harmonics];
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const std::int32_t j = neighbors[p];
                    const int tj = species[j];
                    const double* v = vectors + 3 * p;
                    const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    const double u[3] = {v[0] / r, v[1] / r, v[2] / r};
                    // t_ij = dE_i / dr_ij and t_ji = dE_j / dr_ji at r_ji = -r_ij
                    double t_ij[3] = {0.0, 0.0, 0.0}, t_ji[3] = {0.0, 0.0, 0.0};
                    const double rc_r = 0.5 * (model.rc_radial[ti] + model.rc_radial[tj]);
                    if (r < rc_r) {
                        chebyshev_basis(r, rc_r, kr - 1, f.data(), fp.data());
                        contract(&model.c_radial[static_cast<std::size_t>(ti * types + tj) * nr * kr],
                                 nr, kr, fp.data(), gp_ij.data());
                        contract(&model.c_radial[static_cast<std::size_t>(tj * types + ti) * nr * kr],
                                 nr, kr, fp.data(), gp_ji.data());
                        double a = 0.0, b = 0.0;
                        for (int k = 0; k < nr; ++k) {
                            a += radial_slope[i * nr + k] * gp_ij[k];
                            b += radial_slope[static_cast<std::size_t>(j) * nr + k] * gp_ji[k];
                        }
                        for (int c = 0; c < 3; ++c) {
                            t_ij[c] += a * u[c];
                            t_ji[c] -= b * u[c];
                        }
                    }
                    const double rc_a = 0.5 * (model.rc_angular[ti] + model.rc_angular[tj]);
                    if (r < rc_a) {
                        chebyshev_basis(r, rc_a, ka - 1, f.data(), fp.data());
                        const double* c_ij =
                            &model.c_angular[static_cast<std::size_t>(ti * types + tj) * na * ka];
                        const double* c_ji =
                            &model.c_angular[static_cast<std::size_t>(tj * types + ti) * na * ka];
                        contract(c_ij, na, ka, f.data(), g_ij.data());
                        contract(c_ij, na, ka, fp.data(), gp_ij.data());
                        contract(c_ji, na, ka, f.data(), g_ji.data());
                        contract(c_ji, na, ka, fp.data(), gp_ji.data());
                        evaluate_harmonics(model, u, r, y.data(), gradient.data());
                        const double* gj =
                            &angular_slope[static_cast<std::size_t>(j) * na * harmonics];
                        for (int k = 0; k < na; ++k) {
                            const double* gik = gi + k * harmonics;
                            const double* gjk = gj + k * harmonics;
                            double a_i = 0.0, a_j = 0.0;
                            double b_i[3] = {0.0, 0.0, 0.0}, b_j[3] = {0.0, 0.0, 0.0};
                            for (int h = 0; h < harmonics; ++h) {
                                const double sj = parity[h] * gjk[h];
                                a_i += gik[h] * y[h];
                                a_j += sj * y[h];
                                for (int c = 0; c < 3; ++c) {
                                    b_i[c] += gik[h] * gradient[3 * h + c];
                                    b_j[c] += sj * gradient[3 * h + c];
                                }
                            }
                            for (int c = 0; c < 3; ++c) {
                                t_ij[c] += gp_ij[k] * a_i * u[c] + g_ij[k] * b_i[c];
                                t_ji[c] -= gp_ji[k] * a_j * u[c] + g_ji[k] * b_j[c];
                            }
                        }
                        if (zbl) {
                            double e, de;
                            zbl_pair(&model.zbl[static_cast<std::size_t>(ti * types + tj) * 12], r,
                                     e, de);
                            for (int c = 0; c < 3; ++c) {
                                t_ij[c] += 0.5 * de * u[c];
                                t_ji[c] -= 0.5 * de * u[c];
                            }
                        }
                    }
                    for (int a = 0; a < 3; ++a) {
                        force[a] += t_ij[a] - t_ji[a];
                        for (int b = 0; b < 3; ++b) {
                            w[3 * a + b] += v[a] * t_ji[b];
                        }
                    }
                }
                for (int a = 0; a < 3; ++a) {
                    forces[3 * i + a] = force[a];
                }
                if (virials != nullptr) {
                    for (int k = 0; k < 9; ++k) {
                        virials[9 * i + k] = w[k];
                    }
                }
            }
        },
        control);
}

}  // namespace mdapy::calculators

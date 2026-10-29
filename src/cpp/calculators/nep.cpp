// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/nep.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "neighbor/frame.h"
#include "neighbor/frames.h"

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

// At most this many radial functions, or Chebyshev polynomials, of each
// kind (GPUMD allows 20).
constexpr int kMaxFunctions = 32;

// g_n = sum_k c[n][k] f_k and, unless gp is null, gp_n = sum_k c[n][k]
// fp_k for n < count, from c swapped to [k][n]: each sum runs over k in
// order, across n at once.
void contract(const double* c, int count, int k_count, const double* f, const double* fp,
              double* g, double* gp) {
    double sum[kMaxFunctions] = {}, sum_p[kMaxFunctions] = {};
    if (gp == nullptr) {
        for (int k = 0; k < k_count; ++k) {
            const double* row = c + k * count;
            for (int n = 0; n < count; ++n) {
                sum[n] += row[n] * f[k];
            }
        }
        std::copy(sum, sum + count, g);
        return;
    }
    for (int k = 0; k < k_count; ++k) {
        const double* row = c + k * count;
        for (int n = 0; n < count; ++n) {
            sum[n] += row[n] * f[k];
            sum_p[n] += row[n] * fp[k];
        }
    }
    std::copy(sum, sum + count, g);
    std::copy(sum_p, sum_p + count, gp);
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

// For each pair p = (i, j, s) of a full neighbor list, the pair (j, i, -s).
std::vector<std::int64_t> reverse_pairs(std::int64_t n, const std::int64_t* offsets,
                                        const index_t* neighbors, const std::int8_t* shifts,
                                        const Control& control) {
    const std::int64_t count = offsets[n];
    // each atom's pairs ordered by neighbor and shift
    std::vector<std::int64_t> order(static_cast<std::size_t>(count));
    const auto before = [&](std::int64_t a, std::int64_t b) {
        if (neighbors[a] != neighbors[b]) {
            return neighbors[a] < neighbors[b];
        }
        for (int e = 0; e < 3; ++e) {
            if (shifts[3 * a + e] != shifts[3 * b + e]) {
                return shifts[3 * a + e] < shifts[3 * b + e];
            }
        }
        return false;
    };
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                std::iota(order.begin() + offsets[i], order.begin() + offsets[i + 1], offsets[i]);
                std::sort(order.begin() + offsets[i], order.begin() + offsets[i + 1], before);
            }
        },
        control);
    std::vector<std::int64_t> reverse(static_cast<std::size_t>(count), -1);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const std::int64_t j = neighbors[p];
                    const std::int64_t key[4] = {i, -shifts[3 * p], -shifts[3 * p + 1],
                                                 -shifts[3 * p + 2]};
                    const auto below = [&](std::int64_t q, const std::int64_t* k) {
                        const std::int64_t mine[4] = {neighbors[q], shifts[3 * q],
                                                      shifts[3 * q + 1], shifts[3 * q + 2]};
                        return std::lexicographical_compare(mine, mine + 4, k, k + 4);
                    };
                    const auto first = order.begin() + offsets[j];
                    const auto last = order.begin() + offsets[j + 1];
                    const auto found = std::lower_bound(first, last, key, below);
                    if (found != last && neighbors[*found] == i &&
                        shifts[3 * *found] == key[1] && shifts[3 * *found + 1] == key[2] &&
                        shifts[3 * *found + 2] == key[3]) {
                        reverse[p] = *found;
                    }
                }
            }
        },
        control);
    if (std::find(reverse.begin(), reverse.end(), -1) != reverse.end()) {
        throw std::invalid_argument("the neighbor list is not symmetric");
    }
    return reverse;
}

}  // namespace

void NepModel::prepare() {
    const auto valid = [](int size) { return size >= 0 && size < kMaxFunctions; };
    if (types < 1 || l_max < 1 || l_max > 8 || neurons < 1 || !valid(n_radial) ||
        !valid(n_angular) || !valid(k_radial) || !valid(k_angular) ||
        (four_body && l_max < 2) || (five_body && !four_body)) {
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
    const auto swap = [](const std::vector<double>& a, std::size_t blocks, std::size_t rows,
                         std::size_t columns) {
        std::vector<double> b(a.size());
        for (std::size_t z = 0; z < blocks; ++z) {
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < columns; ++c) {
                    b[(z * columns + c) * rows + r] = a[(z * rows + r) * columns + c];
                }
            }
        }
        return b;
    };
    w0_swapped = swap(w0, t, neurons, dim);
    c_radial_swapped = swap(c_radial, t * t, n_radial + 1, k_radial + 1);
    c_angular_swapped = swap(c_angular, t * t, n_angular + 1, k_angular + 1);
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
                 const double* positions, const double* matrix, const std::int64_t* offsets,
                 const index_t* neighbors, const std::int8_t* shifts, double* energies,
                 double* forces, double* virials, double* descriptors, const Control& control) {
    // the vector of pair p of atom i, as NeighborList.vectors gives it
    const auto vector_of = [&](std::int64_t i, std::int64_t p, double* v) {
        neighbor::pair_vector(positions + 3 * i,
                              positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                              shifts + 3 * p, matrix, v);
    };
    const int types = model.types, dim = model.dim, harmonics = model.harmonic_count();
    const int nr = model.n_radial + 1, na = model.n_angular + 1;
    const int kr = model.k_radial + 1, ka = model.k_angular + 1;
    const int l_max = model.l_max, neurons = model.neurons;
    const int k_size = std::max(kr, ka), n_size = std::max(nr, na);
    const bool zbl = !model.zbl.empty();
    // Without forces, only the descriptors.
    const bool derivatives = forces != nullptr;
    // Each atom in one pass: its descriptors, energy and the derivatives
    // t_p = dE_i / dr_p of its energy by the vector of each of its pairs p.
    // What a pair needs for both, its basis functions and harmonics, is kept
    // from the first loop over the pairs to the second.
    const std::int64_t count = offsets[n];
    std::vector<double> derivative(3 * static_cast<std::size_t>(count));
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> f(k_size), fp(k_size), g(n_size);
            std::vector<double> s(static_cast<std::size_t>(na) * harmonics), q(dim), slope(dim);
            std::vector<double> angular_slope(static_cast<std::size_t>(na) * harmonics);
            std::vector<double> input(static_cast<std::size_t>(neurons));
            std::vector<double> along(static_cast<std::size_t>(harmonics));
            std::vector<double> across(static_cast<std::size_t>(harmonics));
            // per pair of the atom: the radial gp (nr), the angular g and gp
            // (na each), the harmonics and their gradients, the ZBL slope
            std::vector<double> radial_gp, angular_g, angular_gp, y, gradient, zbl_slope;
            for (std::int64_t i = begin; i < end; ++i) {
                const int ti = species[i];
                const std::int64_t first = offsets[i];
                const auto pairs = static_cast<std::size_t>(offsets[i + 1] - first);
                radial_gp.resize(pairs * nr);
                angular_g.resize(pairs * na);
                angular_gp.resize(pairs * na);
                y.resize(pairs * harmonics);
                gradient.resize(pairs * 3 * harmonics);
                zbl_slope.resize(pairs);
                std::fill(q.begin(), q.end(), 0.0);
                std::fill(s.begin(), s.end(), 0.0);
                double repulsion = 0.0;
                for (std::size_t k = 0; k < pairs; ++k) {
                    const std::int64_t p = first + static_cast<std::int64_t>(k);
                    const int tj = species[neighbors[p]];
                    double v[3];
                    vector_of(i, p, v);
                    const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    const double rc_r = 0.5 * (model.rc_radial[ti] + model.rc_radial[tj]);
                    if (r < rc_r) {
                        const double* c_ij =
                            &model.c_radial_swapped[static_cast<std::size_t>(ti * types + tj) * nr * kr];
                        chebyshev_basis(r, rc_r, kr - 1, f.data(), fp.data());
                        contract(c_ij, nr, kr, f.data(), fp.data(), g.data(),
                                 derivatives ? &radial_gp[k * nr] : nullptr);
                        for (int m = 0; m < nr; ++m) {
                            q[m] += g[m];
                        }
                    }
                    const double rc_a = 0.5 * (model.rc_angular[ti] + model.rc_angular[tj]);
                    if (r < rc_a) {
                        const double* c_ij =
                            &model.c_angular_swapped[static_cast<std::size_t>(ti * types + tj) * na * ka];
                        double* gk = &angular_g[k * na];
                        chebyshev_basis(r, rc_a, ka - 1, f.data(), fp.data());
                        contract(c_ij, na, ka, f.data(), fp.data(), gk,
                                 derivatives ? &angular_gp[k * na] : nullptr);
                        const double u[3] = {v[0] / r, v[1] / r, v[2] / r};
                        const double* yk = &y[k * harmonics];
                        evaluate_harmonics(model, u, r, &y[k * harmonics],
                                           derivatives ? &gradient[k * 3 * harmonics] : nullptr);
                        for (int m = 0; m < na; ++m) {
                            for (int h = 0; h < harmonics; ++h) {
                                s[m * harmonics + h] += gk[m] * yk[h];
                            }
                        }
                        if (zbl) {
                            double e;
                            zbl_pair(&model.zbl[static_cast<std::size_t>(ti * types + tj) * 12], r,
                                     e, zbl_slope[k]);
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
                if (!derivatives) {
                    continue;
                }

                // one hidden layer of tanh neurons, the inputs summed over the
                // descriptors in order, across the neurons at once
                double energy = -model.b1;
                std::fill(slope.begin(), slope.end(), 0.0);
                const std::size_t layer = static_cast<std::size_t>(ti) * neurons;
                for (int k = 0; k < neurons; ++k) {
                    input[k] = -model.b0[layer + k];
                }
                for (int d = 0; d < dim; ++d) {
                    const double* w = &model.w0_swapped[(static_cast<std::size_t>(ti) * dim + d) * neurons];
                    for (int k = 0; k < neurons; ++k) {
                        input[k] += w[k] * q[d];
                    }
                }
                for (int k = 0; k < neurons; ++k) {
                    const std::size_t at = layer + k;
                    const double* w = &model.w0[at * dim];
                    const double t = std::tanh(input[k]);
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

                // dE_i / ds_nh of the sums of harmonics
                double* gi = angular_slope.data();
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

                // the derivatives by the pair vectors
                for (std::size_t k = 0; k < pairs; ++k) {
                    const std::int64_t p = first + static_cast<std::int64_t>(k);
                    const int tj = species[neighbors[p]];
                    double v[3];
                    vector_of(i, p, v);
                    const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    const double u[3] = {v[0] / r, v[1] / r, v[2] / r};
                    double* t = &derivative[3 * static_cast<std::size_t>(p)];
                    t[0] = t[1] = t[2] = 0.0;
                    const double rc_r = 0.5 * (model.rc_radial[ti] + model.rc_radial[tj]);
                    if (r < rc_r) {
                        const double* gp = &radial_gp[k * nr];
                        double a = 0.0;
                        for (int m = 0; m < nr; ++m) {
                            a += slope[m] * gp[m];
                        }
                        for (int c = 0; c < 3; ++c) {
                            t[c] += a * u[c];
                        }
                    }
                    const double rc_a = 0.5 * (model.rc_angular[ti] + model.rc_angular[tj]);
                    if (r < rc_a) {
                        // sum_n (g'_n Y_h u + g_n grad Y_h) dE/ds_nh, the sums
                        // over n first, across the harmonics at once
                        const double* gk = &angular_g[k * na];
                        const double* gp = &angular_gp[k * na];
                        const double* yk = &y[k * harmonics];
                        const double* dy = &gradient[k * 3 * harmonics];
                        std::fill(along.begin(), along.end(), 0.0);
                        std::fill(across.begin(), across.end(), 0.0);
                        for (int m = 0; m < na; ++m) {
                            const double* gim = gi + m * harmonics;
                            for (int h = 0; h < harmonics; ++h) {
                                along[h] += gp[m] * gim[h];
                                across[h] += gk[m] * gim[h];
                            }
                        }
                        double a = 0.0, b[3] = {0.0, 0.0, 0.0};
                        for (int h = 0; h < harmonics; ++h) {
                            a += along[h] * yk[h];
                            for (int c = 0; c < 3; ++c) {
                                b[c] += across[h] * dy[3 * h + c];
                            }
                        }
                        for (int c = 0; c < 3; ++c) {
                            t[c] += a * u[c] + b[c];
                        }
                        if (zbl) {
                            for (int c = 0; c < 3; ++c) {
                                t[c] += 0.5 * zbl_slope[k] * u[c];
                            }
                        }
                    }
                }
            }
        },
        control);

    if (!derivatives) {
        return;
    }
    // Forces and virials: each atom gathers the derivatives of its own energy
    // and those of its neighbors' energies by the reverse pairs, r_ji = -r_ij;
    // the virial of a derivative of a neighbor's energy goes to the atom.
    const std::vector<std::int64_t> reverse = reverse_pairs(n, offsets, neighbors, shifts, control);
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                double force[3] = {0.0, 0.0, 0.0};
                double w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    double v[3];
                    vector_of(i, p, v);
                    const double* t_ij = &derivative[3 * static_cast<std::size_t>(p)];
                    const double* t_ji = &derivative[3 * static_cast<std::size_t>(reverse[p])];
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

void compute_nep_frames(const NepModel& model, const neighbor::Frames& frames,
                        const std::int32_t* species, double cutoff, double* energies,
                        double* forces, double* virials, double* descriptors,
                        const Control& control) {
    const auto dim = static_cast<std::size_t>(model.dim);
    neighbor::for_each_frame(
        frames,
        [&](std::int64_t k) {
            const std::int64_t n = frames.atoms(k), s = frames.starts[k];
            const auto atoms = static_cast<std::size_t>(n);
            double* sum = forces != nullptr ? virials + 9 * k : descriptors + dim * static_cast<std::size_t>(k);
            const std::size_t width = forces != nullptr ? 9 : dim;
            std::fill(sum, sum + width, 0.0);
            if (n == 0) {
                if (forces == nullptr) {
                    std::fill(sum, sum + width, std::nan(""));
                }
                return;
            }
            const neighbor::NeighborPairs pairs = frames.pairs(k, cutoff);
            std::vector<double> each(width * atoms);
            compute_nep(model, n, species + s, frames.positions + 3 * s, frames.matrices + 9 * k,
                        pairs.offsets.data(), pairs.neighbors.data(), pairs.shifts.data(),
                        forces != nullptr ? energies + s : nullptr,
                        forces != nullptr ? forces + 3 * s : nullptr,
                        forces != nullptr ? each.data() : nullptr,
                        forces != nullptr ? nullptr : each.data());
            for (std::size_t i = 0; i < atoms; ++i) {
                for (std::size_t c = 0; c < width; ++c) {
                    sum[c] += each[width * i + c];
                }
            }
            if (forces == nullptr) {
                for (std::size_t c = 0; c < width; ++c) {
                    sum[c] /= static_cast<double>(n);
                }
            }
        },
        control);
}

}  // namespace mdapy::calculators

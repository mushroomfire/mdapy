// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/phonon_scattering.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace mdapy::analysis {

namespace {

using Complex = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;

// The Gaussian of x of width sigma.
double gaussian(double x, double sigma) {
    const double z = x / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * kPi));
}

}  // namespace

void three_phonon_rates(const ScatteringInput& in, double* rates, const Control& control) {
    const std::int64_t b = in.modes;
    const std::int64_t t_count = in.temperatures;
    const auto bb = static_cast<std::size_t>(b);
    const std::size_t per_rates = static_cast<std::size_t>(t_count) * bb;
    // the sums of each chunk of wave vectors, added in chunk order so that
    // the result does not depend on the number of threads
    std::vector<double> total = parallel_reduce(
        in.count, 1, std::vector<double>(per_rates, 0.0),
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<double> sums(per_rates, 0.0);
            std::vector<Complex> step1(bb * bb * bb), step2(bb * bb * bb), phi(bb * bb * bb);
            for (std::int64_t k = begin; k < end; ++k) {
                const Complex* v = in.constants + static_cast<std::size_t>(k) * bb * bb * bb;
                const Complex* w1 = in.second + static_cast<std::size_t>(k) * bb * bb;
                const Complex* w2 = in.third + static_cast<std::size_t>(k) * bb * bb;
                // step1[A, m, C] = sum_B v[A, B, C] w1[m, B]
                for (std::size_t a = 0; a < bb; ++a) {
                    for (std::size_t m = 0; m < bb; ++m) {
                        for (std::size_t c = 0; c < bb; ++c) {
                            Complex s = 0.0;
                            for (std::size_t bi = 0; bi < bb; ++bi) {
                                s += v[(a * bb + bi) * bb + c] * w1[m * bb + bi];
                            }
                            step1[(a * bb + m) * bb + c] = s;
                        }
                    }
                }
                // step2[A, m, n] = sum_C step1[A, m, C] w2[n, C]
                for (std::size_t a = 0; a < bb; ++a) {
                    for (std::size_t m = 0; m < bb; ++m) {
                        for (std::size_t n = 0; n < bb; ++n) {
                            Complex s = 0.0;
                            for (std::size_t c = 0; c < bb; ++c) {
                                s += step1[(a * bb + m) * bb + c] * w2[n * bb + c];
                            }
                            step2[(a * bb + m) * bb + n] = s;
                        }
                    }
                }
                // phi[l, m, n] = sum_A conj(w0[l, A]) step2[A, m, n]
                for (std::size_t l = 0; l < bb; ++l) {
                    for (std::size_t mn = 0; mn < bb * bb; ++mn) {
                        Complex s = 0.0;
                        for (std::size_t a = 0; a < bb; ++a) {
                            s += std::conj(in.first[l * bb + a]) * step2[a * bb * bb + mn];
                        }
                        phi[l * bb * bb + mn] = s;
                    }
                }
                const double* f1 = in.frequencies1 + static_cast<std::size_t>(k) * bb;
                const double* f2 = in.frequencies2 + static_cast<std::size_t>(k) * bb;
                for (std::size_t l = 0; l < bb; ++l) {
                    const double w0 = in.frequencies0[l];
                    if (w0 <= in.live) {
                        continue;
                    }
                    for (std::size_t m = 0; m < bb; ++m) {
                        if (f1[m] <= in.live) {
                            continue;
                        }
                        for (std::size_t n = 0; n < bb; ++n) {
                            if (f2[n] <= in.live) {
                                continue;
                            }
                            const std::size_t at = (l * bb + m) * bb + n;
                            const double strength = std::norm(phi[at]) / (w0 * f1[m] * f2[n]);
                            double joined, parted;
                            if (in.sigma > 0.0) {
                                joined = gaussian(w0 - f1[m] - f2[n], in.sigma);
                                parted = gaussian(w0 + f1[m] - f2[n], in.sigma) -
                                         gaussian(w0 - f1[m] + f2[n], in.sigma);
                            } else {
                                const std::size_t flat =
                                    static_cast<std::size_t>(k) * bb * bb * bb + at;
                                joined = in.joined[flat];
                                parted = in.parted[flat];
                            }
                            for (std::int64_t t = 0; t < t_count; ++t) {
                                const std::size_t row =
                                    (static_cast<std::size_t>(t) *
                                         static_cast<std::size_t>(in.count) +
                                     static_cast<std::size_t>(k)) *
                                    bb;
                                const double n1 = in.occupation1[row + m];
                                const double n2 = in.occupation2[row + n];
                                sums[static_cast<std::size_t>(t) * bb + l] +=
                                    strength * ((n1 + n2 + 1.0) * joined + (n1 - n2) * parted);
                            }
                        }
                    }
                }
            }
            return sums;
        },
        [](std::vector<double> a, const std::vector<double>& b2) {
            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i] += b2[i];
            }
            return a;
        },
        control);
    for (std::size_t i = 0; i < per_rates; ++i) {
        rates[i] += total[i];
    }
}

}  // namespace mdapy::analysis

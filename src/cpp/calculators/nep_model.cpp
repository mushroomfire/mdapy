// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The model of a NEP potential: its checks and the tables it derives, apart
// from the computation, so that the GPU part builds it too.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "calculators/nep.h"

namespace mdapy::calculators {

namespace {

constexpr double kPi = 3.14159265358979323846;
// At most this many radial functions, or Chebyshev polynomials, of each
// kind (GPUMD allows 20).
constexpr int kMaxFunctions = 32;

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

}  // namespace mdapy::calculators

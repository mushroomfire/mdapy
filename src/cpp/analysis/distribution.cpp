// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/distribution.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>

#include "neighbor/cell_list.h"

namespace mdapy::analysis {

std::vector<std::int64_t> pair_histogram(const double* positions, std::int64_t n,
                                         const double* matrix, const double* origin,
                                         const bool* pbc, double r_max, int bins,
                                         const std::int32_t* type, int types,
                                         const Control& control) {
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, r_max);
    using Histogram = std::vector<std::int64_t>;
    const std::size_t size = static_cast<std::size_t>(types) * types * bins;
    return parallel_reduce(
        n, 0, Histogram(size, 0),
        [&](std::int64_t begin, std::int64_t end) {
            Histogram local(size, 0);
            for (std::int64_t t = begin; t < end; ++t) {
                const std::int64_t i = grid.atom(t);  // in cell order: integer counts add up the same
                const std::size_t row = static_cast<std::size_t>(type[i]) * types;
                grid.visit(i, [&](std::int32_t j, const std::int32_t*, double r2) {
                    const int bin =
                        std::min(static_cast<int>(std::sqrt(r2) / r_max * bins), bins - 1);
                    ++local[(row + type[j]) * bins + bin];
                });
            }
            return local;
        },
        [](Histogram a, const Histogram& b) {
            for (std::size_t k = 0; k < a.size(); ++k) {
                a[k] += b[k];
            }
            return a;
        },
        control);
}

void structure_factor(const double* positions, std::int64_t n, const double* matrix,
                      const std::int32_t* type, int types, double q_max, int bins,
                      std::vector<double>& sums, std::vector<std::int64_t>& vectors,
                      std::int64_t block, const Control& control) {
    constexpr double kTwoPi = 6.283185307179586476925;
    using Complex = std::complex<double>;
    const double* m = matrix;
    // reciprocal vectors b_i, rows of 2 pi (H^-1)^T
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    double b[3][3];
    for (int i = 0; i < 3; ++i) {
        const double* u = m + 3 * ((i + 1) % 3);
        const double* w = m + 3 * ((i + 2) % 3);
        b[i][0] = kTwoPi * (u[1] * w[2] - u[2] * w[1]) / det;
        b[i][1] = kTwoPi * (u[2] * w[0] - u[0] * w[2]) / det;
        b[i][2] = kTwoPi * (u[0] * w[1] - u[1] * w[0]) / det;
    }
    int reach[3];
    for (int i = 0; i < 3; ++i) {
        const double length = std::sqrt(m[3 * i] * m[3 * i] + m[3 * i + 1] * m[3 * i + 1] +
                                        m[3 * i + 2] * m[3 * i + 2]);
        reach[i] = static_cast<int>(std::ceil(q_max * length / kTwoPi));
    }
    // the vectors of the half space: h > 0, or h = 0 and k > 0, or h = k = 0, l > 0
    struct Vector {
        int h, k, l, bin;
    };
    std::vector<Vector> found;
    for (int h = 0; h <= reach[0]; ++h) {
        for (int k = (h == 0 ? 0 : -reach[1]); k <= reach[1]; ++k) {
            for (int l = (h == 0 && k == 0 ? 1 : -reach[2]); l <= reach[2]; ++l) {
                double q[3];
                for (int e = 0; e < 3; ++e) {
                    q[e] = h * b[0][e] + k * b[1][e] + l * b[2][e];
                }
                const double length = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
                if (length <= q_max) {
                    const int bin = std::min(static_cast<int>(length / q_max * bins), bins - 1);
                    found.push_back({h, k, l, bin});
                }
            }
        }
    }
    // rho_t(q) for every vector and type, summed over blocks of atoms whose
    // phases exp(i h f_d) along each direction are tabulated, so that a vector
    // costs two complex products per atom and the tables stay small
    const int width[3] = {2 * reach[0] + 1, 2 * reach[1] + 1, 2 * reach[2] + 1};
    const auto count = static_cast<std::int64_t>(found.size());
    const std::int64_t row = width[0] + width[1] + width[2];
    if (block <= 0) {
        block = std::max<std::int64_t>(1, std::min<std::int64_t>(n, (std::int64_t{1} << 24) / row));
    }
    std::vector<Complex> rho(static_cast<std::size_t>(count) * types, Complex(0.0, 0.0));
    std::vector<Complex> phase[3];
    for (int d = 0; d < 3; ++d) {
        phase[d].resize(static_cast<std::size_t>(block) * width[d]);
    }
    for (std::int64_t first = 0; first < n; first += block) {
        const std::int64_t size = std::min(block, n - first);
        parallel_for(
            size, 0,
            [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t a = begin; a < end; ++a) {
                    const double* x = positions + 3 * (first + a);
                    for (int d = 0; d < 3; ++d) {
                        // q . r = h f_1 + k f_2 + l f_3 with f_d = r . b_d
                        const double f = x[0] * b[d][0] + x[1] * b[d][1] + x[2] * b[d][2];
                        Complex* row_d = &phase[d][static_cast<std::size_t>(a) * width[d]];
                        for (int h = -reach[d]; h <= reach[d]; ++h) {
                            row_d[h + reach[d]] = std::polar(1.0, h * f);
                        }
                    }
                }
            },
            control);
        parallel_for(
            count, 0,
            [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t v = begin; v < end; ++v) {
                    const Vector& q = found[static_cast<std::size_t>(v)];
                    Complex* sum = &rho[static_cast<std::size_t>(v) * types];
                    for (std::int64_t a = 0; a < size; ++a) {
                        sum[type[first + a]] +=
                            phase[0][static_cast<std::size_t>(a) * width[0] + q.h + reach[0]] *
                            phase[1][static_cast<std::size_t>(a) * width[1] + q.k + reach[1]] *
                            phase[2][static_cast<std::size_t>(a) * width[2] + q.l + reach[2]];
                    }
                }
            },
            control);
    }
    // the products by pair of types, summed in bins
    const std::size_t pairs = static_cast<std::size_t>(types) * types;
    using Partial = std::pair<std::vector<double>, std::vector<std::int64_t>>;
    Partial total = parallel_reduce(
        count, 0, Partial(std::vector<double>(pairs * bins, 0.0), std::vector<std::int64_t>(bins, 0)),
        [&](std::int64_t begin, std::int64_t end) {
            Partial local(std::vector<double>(pairs * bins, 0.0), std::vector<std::int64_t>(bins, 0));
            for (std::int64_t v = begin; v < end; ++v) {
                const Vector& q = found[static_cast<std::size_t>(v)];
                const Complex* sum = &rho[static_cast<std::size_t>(v) * types];
                for (int a = 0; a < types; ++a) {
                    for (int c = 0; c < types; ++c) {
                        local.first[(static_cast<std::size_t>(a) * types + c) * bins + q.bin] +=
                            (sum[a] * std::conj(sum[c])).real();
                    }
                }
                ++local.second[q.bin];
            }
            return local;
        },
        [](Partial a, const Partial& c) {
            for (std::size_t k = 0; k < a.first.size(); ++k) {
                a.first[k] += c.first[k];
            }
            for (std::size_t k = 0; k < a.second.size(); ++k) {
                a.second[k] += c.second[k];
            }
            return a;
        },
        control);
    sums = std::move(total.first);
    vectors = std::move(total.second);
}

}  // namespace mdapy::analysis

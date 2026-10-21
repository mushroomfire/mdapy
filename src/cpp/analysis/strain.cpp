// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/strain.h"

#include <cmath>
#include <limits>

namespace mdapy::analysis {
namespace {

// The inverse of a 3 x 3 matrix (row major); false when it is singular
// relative to its size.
bool invert(const double* m, double* inverse) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    double scale = 0.0;
    for (int k = 0; k < 9; ++k) {
        scale = std::fmax(scale, std::fabs(m[k]));
    }
    if (!(std::fabs(det) > 1e-12 * scale * scale * scale)) {
        return false;
    }
    inverse[0] = (m[4] * m[8] - m[5] * m[7]) / det;
    inverse[1] = (m[2] * m[7] - m[1] * m[8]) / det;
    inverse[2] = (m[1] * m[5] - m[2] * m[4]) / det;
    inverse[3] = (m[5] * m[6] - m[3] * m[8]) / det;
    inverse[4] = (m[0] * m[8] - m[2] * m[6]) / det;
    inverse[5] = (m[2] * m[3] - m[0] * m[5]) / det;
    inverse[6] = (m[3] * m[7] - m[4] * m[6]) / det;
    inverse[7] = (m[1] * m[6] - m[0] * m[7]) / det;
    inverse[8] = (m[0] * m[4] - m[1] * m[3]) / det;
    return true;
}

// row vector f times the 3 x 3 matrix h
void times(const double* f, const double* h, double* out) {
    for (int e = 0; e < 3; ++e) {
        out[e] = f[0] * h[e] + f[1] * h[3 + e] + f[2] * h[6 + e];
    }
}

}  // namespace

void atomic_strain(std::int64_t n, const std::int64_t* offsets, const std::int32_t* neighbors,
                   const std::int32_t* shifts, const double* reference, const double* displacement,
                   const double* reference_matrix, const double* matrix, double* strain,
                   double* nonaffine, const Control& control) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                // V = sum d0 d0^T, W = sum d d0^T; F = W V^-1
                double v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const std::int64_t j = neighbors[p];
                    double f0[3], f[3], d0[3], d[3];
                    for (int e = 0; e < 3; ++e) {
                        f0[e] = reference[3 * j + e] + shifts[3 * p + e] - reference[3 * i + e];
                        f[e] = f0[e] + displacement[3 * j + e] - displacement[3 * i + e];
                    }
                    times(f0, reference_matrix, d0);
                    times(f, matrix, d);
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            v[3 * a + b] += d0[a] * d0[b];
                            w[3 * a + b] += d[a] * d0[b];
                        }
                    }
                }
                double inverse[9], gradient[9];
                if (!invert(v, inverse)) {
                    for (int k = 0; k < 9; ++k) {
                        strain[9 * i + k] = nan;
                    }
                    nonaffine[i] = nan;
                    continue;
                }
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        gradient[3 * a + b] = w[3 * a] * inverse[b] + w[3 * a + 1] * inverse[3 + b] +
                                              w[3 * a + 2] * inverse[6 + b];
                    }
                }
                // Green-Lagrange strain (F^T F - I) / 2
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        double sum = 0.0;
                        for (int c = 0; c < 3; ++c) {
                            sum += gradient[3 * c + a] * gradient[3 * c + b];
                        }
                        strain[9 * i + 3 * a + b] = 0.5 * (sum - (a == b ? 1.0 : 0.0));
                    }
                }
                // sum of |d - F d0|^2
                double residual = 0.0;
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const std::int64_t j = neighbors[p];
                    double f0[3], f[3], d0[3], d[3];
                    for (int e = 0; e < 3; ++e) {
                        f0[e] = reference[3 * j + e] + shifts[3 * p + e] - reference[3 * i + e];
                        f[e] = f0[e] + displacement[3 * j + e] - displacement[3 * i + e];
                    }
                    times(f0, reference_matrix, d0);
                    times(f, matrix, d);
                    for (int a = 0; a < 3; ++a) {
                        const double mapped = gradient[3 * a] * d0[0] + gradient[3 * a + 1] * d0[1] +
                                              gradient[3 * a + 2] * d0[2];
                        residual += (d[a] - mapped) * (d[a] - mapped);
                    }
                }
                nonaffine[i] = residual;
            }
        },
        control);
}

}  // namespace mdapy::analysis

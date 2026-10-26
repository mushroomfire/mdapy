// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/strain.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/buffer.h"
#include "neighbor/cell_list.h"

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

namespace {

// The fractional coordinates s0 of atom i in the reference box, and its
// displacement from them in fractional coordinates of the current box, by
// the nearest image along periodic directions.
void reference_and_step(const StrainInput& in, std::int64_t i, double* s0, double* step) {
    const double* x0 = in.reference + 3 * i;
    const double* x = in.positions + 3 * i;
    const double r0[3] = {x0[0] - in.reference_origin[0], x0[1] - in.reference_origin[1],
                          x0[2] - in.reference_origin[2]};
    const double r[3] = {x[0] - in.origin[0], x[1] - in.origin[1], x[2] - in.origin[2]};
    const double* g0 = in.reference_inverse;
    const double* g = in.inverse;
    for (int e = 0; e < 3; ++e) {
        s0[e] = r0[0] * g0[e] + r0[1] * g0[3 + e] + r0[2] * g0[6 + e];
        const double f = r[0] * g[e] + r[1] * g[3 + e] + r[2] * g[6 + e];
        step[e] = f - s0[e];
        if (in.pbc[e]) {
            step[e] -= std::nearbyint(step[e]);
        }
    }
}

// The atomic strain of the atom at row i of the reference fractional
// coordinates and the displacements, from its count reference neighbors at
// rows js with their periodic images shift (count x 3), in the order given,
// written to row o of the results; vectors holds the neighbor vectors for
// the non-affine displacement.
template <class Index, class Shift>
void strain_of(std::int64_t i, std::int64_t o, std::int64_t count, const Index* js,
               const Shift* shifts, const double* reference, const double* displacement,
               const StrainInput& in, const StrainOutput& out, std::vector<double>& vectors) {
    const bool nonaffine = out.nonaffine != nullptr;
    if (nonaffine) {
        vectors.resize(6 * static_cast<std::size_t>(count));
    }
    // V = sum d0 d0^T, W = sum d d0^T; F = W V^-1
    double v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (std::int64_t p = 0; p < count; ++p) {
        const std::int64_t j = js[p];
        double f0[3], f[3], d0[3], d[3];
        for (int e = 0; e < 3; ++e) {
            f0[e] = reference[3 * j + e] + shifts[3 * p + e] - reference[3 * i + e];
            f[e] = f0[e] + displacement[3 * j + e] - displacement[3 * i + e];
        }
        times(f0, in.reference_matrix, d0);
        times(f, in.matrix, d);
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                v[3 * a + b] += d0[a] * d0[b];
                w[3 * a + b] += d[a] * d0[b];
            }
        }
        if (nonaffine) {
            std::copy(d0, d0 + 3, &vectors[6 * p]);
            std::copy(d, d + 3, &vectors[6 * p + 3]);
        }
    }
    double inverse[9], gradient[9], e[9];
    if (!invert(v, inverse)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (out.strain != nullptr) {
            std::fill(out.strain + 9 * o, out.strain + 9 * o + 9, nan);
        }
        out.shear[o] = out.volumetric[o] = nan;
        if (nonaffine) {
            out.nonaffine[o] = nan;
        }
        return;
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
            e[3 * a + b] = 0.5 * (sum - (a == b ? 1.0 : 0.0));
        }
    }
    if (out.strain != nullptr) {
        std::copy(e, e + 9, out.strain + 9 * o);
    }
    out.shear[o] = std::sqrt(e[1] * e[1] + e[2] * e[2] + e[5] * e[5] +
                             ((e[0] - e[4]) * (e[0] - e[4]) + (e[4] - e[8]) * (e[4] - e[8]) +
                              (e[0] - e[8]) * (e[0] - e[8])) /
                                 6.0);
    out.volumetric[o] = (e[0] + e[4] + e[8]) / 3.0;
    if (nonaffine) {
        // sum of |d - F d0|^2
        double residual = 0.0;
        for (std::int64_t p = 0; p < count; ++p) {
            const double* d0 = &vectors[6 * p];
            const double* d = d0 + 3;
            for (int a = 0; a < 3; ++a) {
                const double mapped = gradient[3 * a] * d0[0] + gradient[3 * a + 1] * d0[1] +
                                      gradient[3 * a + 2] * d0[2];
                residual += (d[a] - mapped) * (d[a] - mapped);
            }
        }
        out.nonaffine[o] = residual;
    }
}

}  // namespace

void atomic_strain_search(double cutoff, const StrainInput& input, const StrainOutput& output,
                          const Control& control) {
    if (input.n == 0) {
        return;
    }
    const neighbor::CellGrid grid(input.reference, input.n, input.reference_matrix,
                                  input.reference_origin, input.pbc, cutoff);
    // the reference coordinates and the displacements in the order of the
    // slots, where the neighbors of an atom lie close together
    const auto n = static_cast<std::size_t>(input.n);
    Buffer<double> reference(3 * n), moved(3 * n);
    parallel_for(
        input.n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t t = begin; t < end; ++t) {
                reference_and_step(input, grid.atom(t), &reference[3 * t], &moved[3 * t]);
            }
        },
        stage(control, 0, 2));
    parallel_for(
        input.n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<std::int64_t> slots;
            std::vector<std::int32_t> shifts;
            std::vector<double> vectors;
            for (std::int64_t t = begin; t < end; ++t) {
                slots.clear();
                shifts.clear();
                grid.visit_slots(t, [&](std::int64_t s, const std::int32_t* shift) {
                    slots.push_back(s);
                    shifts.insert(shifts.end(), shift, shift + 3);
                });
                strain_of(t, grid.atom(t), static_cast<std::int64_t>(slots.size()), slots.data(),
                          shifts.data(), reference.data(), moved.data(), input, output, vectors);
            }
        },
        stage(control, 1, 2));
}

void atomic_strain_list(const std::int64_t* offsets, const index_t* neighbors,
                        const std::int8_t* shifts, const double* distances, double cutoff,
                        const StrainInput& input, const StrainOutput& output,
                        const Control& control) {
    if (input.n == 0) {
        return;
    }
    const auto n = static_cast<std::size_t>(input.n);
    Buffer<double> reference(3 * n), moved(3 * n);
    parallel_for(
        input.n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                reference_and_step(input, i, &reference[3 * i], &moved[3 * i]);
            }
        },
        stage(control, 0, 2));
    parallel_for(
        input.n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<index_t> js;
            std::vector<std::int8_t> kept;
            std::vector<double> vectors;
            for (std::int64_t i = begin; i < end; ++i) {
                js.clear();
                kept.clear();
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    if (distances[p] < cutoff) {
                        js.push_back(neighbors[p]);
                        kept.insert(kept.end(), shifts + 3 * p, shifts + 3 * p + 3);
                    }
                }
                strain_of(i, i, static_cast<std::int64_t>(js.size()), js.data(), kept.data(),
                          reference.data(), moved.data(), input, output, vectors);
            }
        },
        stage(control, 1, 2));
}

}  // namespace mdapy::analysis

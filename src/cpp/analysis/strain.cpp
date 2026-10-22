// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/strain.h"

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

// The displacement of every atom from the reference, in fractional
// coordinates of the current box, by the nearest image along periodic
// directions (n x 3).
Buffer<double> displacements(const StrainInput& in, const Control& control) {
    Buffer<double> d(3 * static_cast<std::size_t>(in.n));
    parallel_for(
        in.n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                const double* x = in.positions + 3 * i;
                const double r[3] = {x[0] - in.origin[0], x[1] - in.origin[1], x[2] - in.origin[2]};
                for (int e = 0; e < 3; ++e) {
                    const double f = r[0] * in.inverse[e] + r[1] * in.inverse[3 + e] +
                                     r[2] * in.inverse[6 + e];
                    double step = f - in.reference[3 * i + e];
                    if (in.pbc[e]) {
                        step -= std::nearbyint(step);
                    }
                    d[3 * static_cast<std::size_t>(i) + e] = step;
                }
            }
        },
        stage(control, 0, 2));
    return d;
}

// The atomic strain of the atom at row i of the reference coordinates and
// the displacements from its count reference neighbors at rows js with
// their periodic images shift (count x 3), in the order given, written to
// row o of the results.
template <class Index, class Shift>
void strain_of(std::int64_t i, std::int64_t o, std::int64_t count, const Index* js,
               const Shift* shifts, const StrainInput& in, const double* displacement,
               const StrainOutput& out) {
    const auto vectors = [&](std::int64_t p, double* d0, double* d) {
        const std::int64_t j = js[p];
        double f0[3], f[3];
        for (int e = 0; e < 3; ++e) {
            f0[e] = in.reference[3 * j + e] + shifts[3 * p + e] - in.reference[3 * i + e];
            f[e] = f0[e] + displacement[3 * j + e] - displacement[3 * i + e];
        }
        times(f0, in.reference_matrix, d0);
        times(f, in.matrix, d);
    };
    // V = sum d0 d0^T, W = sum d d0^T; F = W V^-1
    double v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, w[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (std::int64_t p = 0; p < count; ++p) {
        double d0[3], d[3];
        vectors(p, d0, d);
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                v[3 * a + b] += d0[a] * d0[b];
                w[3 * a + b] += d[a] * d0[b];
            }
        }
    }
    double* e = out.strain + 9 * o;
    double inverse[9], gradient[9];
    if (!invert(v, inverse)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (int k = 0; k < 9; ++k) {
            e[k] = nan;
        }
        out.shear[o] = out.volumetric[o] = out.nonaffine[o] = nan;
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
    out.shear[o] = std::sqrt(e[1] * e[1] + e[2] * e[2] + e[5] * e[5] +
                             ((e[0] - e[4]) * (e[0] - e[4]) + (e[4] - e[8]) * (e[4] - e[8]) +
                              (e[0] - e[8]) * (e[0] - e[8])) /
                                 6.0);
    out.volumetric[o] = (e[0] + e[4] + e[8]) / 3.0;
    // sum of |d - F d0|^2
    double residual = 0.0;
    for (std::int64_t p = 0; p < count; ++p) {
        double d0[3], d[3];
        vectors(p, d0, d);
        for (int a = 0; a < 3; ++a) {
            const double mapped =
                gradient[3 * a] * d0[0] + gradient[3 * a + 1] * d0[1] + gradient[3 * a + 2] * d0[2];
            residual += (d[a] - mapped) * (d[a] - mapped);
        }
    }
    out.nonaffine[o] = residual;
}

}  // namespace

void atomic_strain_search(const double* reference_positions, const double* reference_origin,
                          double cutoff, const StrainInput& input, const StrainOutput& output,
                          const Control& control) {
    if (input.n == 0) {
        return;
    }
    const Buffer<double> displacement = displacements(input, control);
    const neighbor::CellGrid grid(reference_positions, input.n, input.reference_matrix,
                                  reference_origin, input.pbc, cutoff);
    // the reference coordinates and the displacements in the order of the
    // slots, where the neighbors of an atom lie close together
    const auto n = static_cast<std::size_t>(input.n);
    Buffer<double> reference(3 * n), moved(3 * n);
    parallel_for(input.n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t t = begin; t < end; ++t) {
            const std::int64_t i = grid.atom(t);
            for (int e = 0; e < 3; ++e) {
                reference[3 * t + e] = input.reference[3 * i + e];
                moved[3 * t + e] = displacement[3 * i + e];
            }
        }
    });
    StrainInput slotted = input;
    slotted.reference = reference.data();
    parallel_for(
        input.n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<std::int64_t> slots;
            std::vector<std::int32_t> shifts;
            for (std::int64_t t = begin; t < end; ++t) {
                slots.clear();
                shifts.clear();
                grid.visit_slots(t, [&](std::int64_t s, const std::int32_t* shift) {
                    slots.push_back(s);
                    shifts.insert(shifts.end(), shift, shift + 3);
                });
                strain_of(t, grid.atom(t), static_cast<std::int64_t>(slots.size()), slots.data(),
                          shifts.data(), slotted, moved.data(), output);
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
    const Buffer<double> displacement = displacements(input, control);
    parallel_for(
        input.n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<index_t> js;
            std::vector<std::int8_t> kept;
            for (std::int64_t i = begin; i < end; ++i) {
                js.clear();
                kept.clear();
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    if (distances[p] < cutoff) {
                        js.push_back(neighbors[p]);
                        kept.insert(kept.end(), shifts + 3 * p, shifts + 3 * p + 3);
                    }
                }
                strain_of(i, i, static_cast<std::int64_t>(js.size()), js.data(), kept.data(), input,
                          displacement.data(), output);
            }
        },
        stage(control, 1, 2));
}

}  // namespace mdapy::analysis

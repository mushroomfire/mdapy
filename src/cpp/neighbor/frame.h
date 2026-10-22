// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// The box as the neighbor searches see it, and the vector of a pair.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mdapy::neighbor {

// The vector r = x_j - x_i + shift . H of a pair, H with the lattice vectors
// as rows, and its squared length. Every search and list computes it in this
// order, so a distance has the same bits whichever code finds the pair.
template <class Shift>
inline double pair_vector(const double* xi, const double* xj, const Shift* shift,
                          const double* h, double* r) {
    double r2 = 0.0;
    for (int e = 0; e < 3; ++e) {
        r[e] = xj[e] - xi[e] + shift[0] * h[e] + shift[1] * h[3 + e] + shift[2] * h[6 + e];
        r2 += r[e] * r[e];
    }
    return r2;
}

// Throws std::overflow_error when a shift does not fit the int8 of a list.
inline void check_shift(const std::int32_t* shift) {
    for (int e = 0; e < 3; ++e) {
        if (shift[e] < -127 || shift[e] > 127) {
            throw std::overflow_error(
                "a periodic image shift of " + std::to_string(shift[e]) +
                " is outside the int8 range [-127, 127]; wrap the positions into the box "
                "(Box.wrap) or use a box thicker than 1/127 of the cutoff");
        }
    }
}

// The lattice vectors, the inverse matrix and the periodic directions.
struct Frame {
    double h[9];          // rows a, b, c
    double inverse[9];    // fractional coordinate d of x is (x - origin) . inverse[3d:3d+3]
    double origin[3];
    double thickness[3];  // distance between the faces across each lattice direction
    double lengths[3];    // largest component of each lattice vector, for rounding bounds
    bool pbc[3];

    // Throws std::invalid_argument for a singular box.
    Frame(const double* matrix, const double* box_origin, const bool* periodic) {
        const double* a = matrix;
        const double* b = matrix + 3;
        const double* c = matrix + 6;
        // rows of the adjugate: b x c, c x a, a x b
        const double cross[9] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                                 b[0] * c[1] - b[1] * c[0], c[1] * a[2] - c[2] * a[1],
                                 c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0],
                                 a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                 a[0] * b[1] - a[1] * b[0]};
        const double det = a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2];
        if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
            throw std::invalid_argument("the box is singular");
        }
        for (int d = 0; d < 3; ++d) {
            const double* n = cross + 3 * d;
            thickness[d] = std::abs(det) / std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            lengths[d] = 0.0;
            for (int e = 0; e < 3; ++e) {
                h[3 * d + e] = matrix[3 * d + e];
                inverse[3 * d + e] = n[e] / det;
                lengths[d] = std::max(lengths[d], std::abs(matrix[3 * d + e]));
            }
            origin[d] = box_origin[d];
            pbc[d] = periodic[d];
        }
    }

    // The fractional coordinates f of x, wrapped into [0, 1] along periodic
    // directions by w periodic images: x - w . H is in the box. Throws
    // std::invalid_argument for a position that is not finite or is more
    // than 1e9 box lengths away.
    void wrap(const double* x, double* f, std::int32_t* w) const {
        const double r[3] = {x[0] - origin[0], x[1] - origin[1], x[2] - origin[2]};
        for (int d = 0; d < 3; ++d) {
            f[d] = r[0] * inverse[3 * d] + r[1] * inverse[3 * d + 1] + r[2] * inverse[3 * d + 2];
            if (!(std::abs(f[d]) < 1e9)) {
                throw std::invalid_argument(
                    "positions must be finite and within 1e9 box lengths of the box");
            }
            w[d] = 0;
            if (pbc[d]) {
                const double k = std::floor(f[d]);
                f[d] -= k;
                w[d] = static_cast<std::int32_t>(k);
            }
        }
    }

    // The image y = x - w . H of x.
    void image(const double* x, const std::int32_t* w, double* y) const {
        for (int e = 0; e < 3; ++e) {
            y[e] = x[e] - (w[0] * h[e] + w[1] * h[3 + e] + w[2] * h[6 + e]);
        }
    }
};

}  // namespace mdapy::neighbor

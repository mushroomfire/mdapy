// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Positions mapped into a box.

#pragma once

#include <cmath>
#include <cstdint>

#include "core/parallel.h"

namespace mdapy {

// Writes the n positions moved into the box along its periodic directions and
// their image flags, positions = wrapped + image . matrix up to rounding.
// matrix and inverse: 3 x 3, the lattice vectors as rows; origin, pbc: 3. Whole
// lattice vectors are subtracted, which keeps the precision of the input.
inline void wrap(const double* positions, std::int64_t n, const double* matrix,
                 const double* inverse, const double* origin, const bool* pbc, double* wrapped,
                 std::int64_t* image, const Control& control = {}) {
    const auto fraction = [&](const double* r, double* f) {
        const double d[3] = {r[0] - origin[0], r[1] - origin[1], r[2] - origin[2]};
        for (int e = 0; e < 3; ++e) {
            f[e] = d[0] * inverse[e] + d[1] * inverse[3 + e] + d[2] * inverse[6 + e];
        }
    };
    const auto subtract = [&](double* r, const std::int64_t* flags) {
        for (int e = 0; e < 3; ++e) {
            r[e] -= static_cast<double>(flags[0]) * matrix[e] +
                    static_cast<double>(flags[1]) * matrix[3 + e] +
                    static_cast<double>(flags[2]) * matrix[6 + e];
        }
    };
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                double* w = wrapped + 3 * i;
                std::int64_t* flags = image + 3 * i;
                double f[3];
                fraction(positions + 3 * i, f);
                for (int e = 0; e < 3; ++e) {
                    w[e] = positions[3 * i + e];
                    // a position that is not finite stays where it is
                    flags[e] = pbc[e] && std::isfinite(f[e])
                                   ? static_cast<std::int64_t>(std::floor(f[e]))
                                   : 0;
                }
                subtract(w, flags);
                // rounding can leave a coordinate just outside [0, 1); fix it once
                fraction(w, f);
                std::int64_t again[3];
                bool any = false;
                for (int e = 0; e < 3; ++e) {
                    again[e] = !pbc[e] ? 0 : f[e] < 0.0 ? -1 : f[e] >= 1.0 ? 1 : 0;
                    any = any || again[e] != 0;
                }
                if (any) {
                    subtract(w, again);
                    for (int e = 0; e < 3; ++e) {
                        flags[e] += again[e];
                    }
                }
            }
        },
        control);
}

}  // namespace mdapy

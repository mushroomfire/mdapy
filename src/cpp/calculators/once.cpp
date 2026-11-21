// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/once.h"

#include "neighbor/frame.h"

namespace mdapy::calculators {

OncePairs once_pairs(std::int64_t n, std::int64_t centers, const std::int64_t* offsets,
                     const index_t* neighbors, const std::int8_t* shifts,
                     const std::int64_t* ids, const double* positions, const double* matrix) {
    OncePairs out;
    out.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    out.neighbors.reserve(static_cast<std::size_t>(offsets[centers] / 2 + 1));
    out.shifts.reserve(3 * static_cast<std::size_t>(offsets[centers] / 2 + 1));
    for (std::int64_t i = 0; i < centers; ++i) {
        for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
            const auto j = static_cast<std::int64_t>(neighbors[p]);
            const std::int8_t* shift = shifts + 3 * p;
            if (!once_first(i, j, shift, centers, ids)) {
                continue;
            }
            double v[3];
            neighbor::pair_vector(positions + 3 * i, positions + 3 * j, shift, matrix, v);
            if (!once_then(i, j, v, centers, ids)) {
                continue;
            }
            out.neighbors.push_back(neighbors[p]);
            out.shifts.insert(out.shifts.end(), shift, shift + 3);
        }
        out.offsets[static_cast<std::size_t>(i) + 1] =
            static_cast<std::int64_t>(out.neighbors.size());
    }
    for (std::int64_t i = centers; i < n; ++i) {
        out.offsets[static_cast<std::size_t>(i) + 1] = out.offsets[static_cast<std::size_t>(i)];
    }
    return out;
}

}  // namespace mdapy::calculators

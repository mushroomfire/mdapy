// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/once.h"

#include <cmath>

#include "neighbor/cell_list.h"
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

OncePairs build_once(const double* positions, std::int64_t n, const double* matrix,
                     const double* origin, const bool* pbc, double cutoff, std::int64_t centers,
                     const std::int64_t* ids, OncePairs* full) {
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, cutoff);
    std::vector<std::int64_t> slot(static_cast<std::size_t>(n));
    for (std::int64_t t = 0; t < n; ++t) slot[static_cast<std::size_t>(grid.atom(t))] = t;
    OncePairs out;
    out.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    if (full != nullptr) full->offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    for (std::int64_t i = 0; i < centers; ++i) {
        grid.visit(slot[static_cast<std::size_t>(i)],
                   [&](index_t neighbor, const std::int32_t* found, double r2) {
                       neighbor::check_shift(found);
                       const std::int8_t shift[3] = {static_cast<std::int8_t>(found[0]),
                                                     static_cast<std::int8_t>(found[1]),
                                                     static_cast<std::int8_t>(found[2])};
                       const double distance = std::sqrt(r2);
                       if (full != nullptr) {
                           full->neighbors.push_back(neighbor);
                           full->shifts.insert(full->shifts.end(), shift, shift + 3);
                           full->distances.push_back(distance);
                       }
                       const auto j = static_cast<std::int64_t>(neighbor);
                       if (!once_first(i, j, shift, centers, ids)) return;
                       if (j >= centers && ids[i] == ids[j]) {
                           double v[3];
                           neighbor::pair_vector(positions + 3 * i, positions + 3 * j, shift,
                                                 matrix, v);
                           if (!once_then(i, j, v, centers, ids)) return;
                       }
                       out.neighbors.push_back(neighbor);
                       out.shifts.insert(out.shifts.end(), shift, shift + 3);
                       out.distances.push_back(distance);
                   });
        out.offsets[static_cast<std::size_t>(i) + 1] = static_cast<std::int64_t>(out.neighbors.size());
        if (full != nullptr) {
            full->offsets[static_cast<std::size_t>(i) + 1] =
                static_cast<std::int64_t>(full->neighbors.size());
        }
    }
    for (std::int64_t i = centers; i < n; ++i) {
        out.offsets[static_cast<std::size_t>(i) + 1] = out.offsets[static_cast<std::size_t>(i)];
        if (full != nullptr) {
            full->offsets[static_cast<std::size_t>(i) + 1] =
                full->offsets[static_cast<std::size_t>(i)];
        }
    }
    return out;
}

}  // namespace mdapy::calculators

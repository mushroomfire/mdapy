// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "neighbor/frames.h"

#include <algorithm>
#include <limits>

namespace mdapy::neighbor {

std::vector<double> closest_pairs(const Frames& frames, const double* radii, double close,
                                  const Control& control) {
    std::vector<double> found(static_cast<std::size_t>(frames.count),
                              std::numeric_limits<double>::infinity());
    for_each_frame(
        frames,
        [&](std::int64_t k) {
            const std::int64_t n = frames.atoms(k);
            if (n < 2) {
                return;
            }
            const double* r = radii + frames.starts[k];
            const double largest = *std::max_element(r, r + n);
            const NeighborPairs pairs = frames.pairs(k, close * 2.0 * largest);
            double best = std::numeric_limits<double>::infinity();
            for (std::int64_t i = 0; i < n; ++i) {
                for (std::int64_t p = pairs.offsets[i]; p < pairs.offsets[i + 1]; ++p) {
                    best = std::min(best, pairs.distances[p] / (r[i] + r[pairs.neighbors[p]]));
                }
            }
            if (best < close) {
                found[static_cast<std::size_t>(k)] = best;
            }
        },
        control);
    return found;
}

}  // namespace mdapy::neighbor

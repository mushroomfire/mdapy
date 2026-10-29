// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Many small frames one after another, such as a training set, computed a
// frame on a thread: the work of a frame, its own parallel loops included,
// runs on the thread that took it, so the result does not depend on the
// threads.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"
#include "neighbor/cell_list.h"

namespace mdapy::neighbor {

// Frame k holds the atoms [starts[k], starts[k + 1]) of positions (n x 3);
// its box has the lattice vectors matrices[9k:9k+9] as rows, the origin
// origins[3k:3k+3] and the periodic directions pbc[3k:3k+3].
struct Frames {
    std::int64_t count = 0;
    const std::int64_t* starts = nullptr;
    const double* positions = nullptr;
    const double* matrices = nullptr;
    const double* origins = nullptr;
    const bool* pbc = nullptr;

    std::int64_t atoms(std::int64_t k) const { return starts[k + 1] - starts[k]; }
    // the pairs of frame k within cutoff, as build finds them for the frame alone
    NeighborPairs pairs(std::int64_t k, double cutoff) const {
        return build(positions + 3 * starts[k], atoms(k), matrices + 9 * k, origins + 3 * k, pbc + 3 * k,
                     cutoff);
    }
};

// Calls work(k) for each frame k, frames on threads.
template <class Work>
void for_each_frame(const Frames& frames, Work&& work, const Control& control = {}) {
    parallel_for(
        frames.count, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t k = begin; k < end; ++k) {
                work(k);
            }
        },
        control);
}

// For each frame, the smallest distance of two of its atoms over the sum of
// their radii (n), where it is below close, else infinity; searched within
// close times twice the largest radius of the frame.
std::vector<double> closest_pairs(const Frames& frames, const double* radii, double close,
                                  const Control& control = {});

}  // namespace mdapy::neighbor

// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Shortest-path rings of a network of bonds: closed paths of bonds through
// distinct atoms in which no two atoms are closer through the network than
// along the ring, the periodic images of the atoms told apart, so that a
// path around the periodic box is no ring.

#pragma once

#include <cstdint>
#include <vector>

#include "core/parallel.h"

namespace mdapy::analysis {

// The bonds of n atoms, each both ways: those of atom i are
// offsets[i]..offsets[i + 1] of other (the other atom) and shifts (its
// periodic image, 3 each).
struct RingBonds {
    std::int64_t n = 0;
    const std::int64_t* offsets = nullptr;
    const std::int64_t* other = nullptr;
    const std::int32_t* shifts = nullptr;
};

// The rings, each from its atom of the smallest index, then on toward the
// smaller of its two neighbors, in the order of that first atom, then of
// size and of their atoms: the atoms of all rings one after the other, the
// image of each relative to the first of its ring (3 each), and where each
// ring starts (rings + 1).
struct Rings {
    std::vector<std::int64_t> atoms;
    std::vector<std::int32_t> shifts;
    std::vector<std::int64_t> starts;
};

Rings find_rings(const RingBonds& bonds, int min_size, int max_size, const Control& control = {});

}  // namespace mdapy::analysis

// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Each pair once among the processes of MPI, as LAMMPS's newton on: the
// first centers atoms of a system are a process's own, the others its ghosts,
// copies of the atoms of other processes or of its own across the box. A pair
// of two centres is computed from the lower of them (an atom and its own image
// from the shift whose first component not zero is positive); a pair of a
// centre and a ghost by the process of the one of their two ids that a hash of
// both picks, which the other process finds too, so that exactly one of them
// computes it and each computes half the pairs across a face, however the ids
// lie in space; a centre and a ghost of itself from the vector whose first
// component not zero is positive. The contributions to the ghosts go back to
// their atoms.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"

namespace mdapy::calculators {

// Whether the process of the atom of id a computes its pair with the atom of
// id b, another: the lower or the higher of the two, as a hash of both says.
inline bool picks(std::int64_t a, std::int64_t b) {
    const auto lo = static_cast<std::uint64_t>(a < b ? a : b);
    const auto hi = static_cast<std::uint64_t>(a < b ? b : a);
    std::uint64_t z = lo + 0x9E3779B97F4A7C15ull * (hi + 1);  // the finalizer of splitmix64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (z & 1u) == 0 ? a < b : a > b;
}

// What decides before the pair vector: false for a pair another row or
// process computes.
inline bool once_first(std::int64_t i, std::int64_t j, const std::int8_t* shift,
                       std::int64_t centers, const std::int64_t* ids) {
    if (j >= centers) {
        return ids[i] == ids[j] || picks(ids[i], ids[j]);
    }
    if (i != j) {
        return i < j;
    }
    for (int e = 0; e < 3; ++e) {
        if (shift[e] != 0) {
            return shift[e] > 0;
        }
    }
    return false;
}

// What decides with the pair vector v = x_j - x_i: an atom and its own ghost.
inline bool once_then(std::int64_t i, std::int64_t j, const double* v, std::int64_t centers,
                      const std::int64_t* ids) {
    if (j < centers || ids[i] != ids[j]) {
        return true;
    }
    for (int e = 0; e < 3; ++e) {
        if (v[e] != 0.0) {
            return v[e] > 0.0;
        }
    }
    return false;
}

// The pairs of a full neighbor list of n atoms that a process computes on one
// thread, those of its first centers atoms that once_first and once_then
// keep, in the order of the list: as a list of its own, the rows of the
// other atoms empty, so that the kernels go through it without asking.
struct OncePairs {
    std::vector<std::int64_t> offsets;  // n + 1
    std::vector<index_t> neighbors;
    std::vector<std::int8_t> shifts;  // 3 per pair
};
OncePairs once_pairs(std::int64_t n, std::int64_t centers, const std::int64_t* offsets,
                     const index_t* neighbors, const std::int8_t* shifts,
                     const std::int64_t* ids, const double* positions, const double* matrix);

}  // namespace mdapy::calculators

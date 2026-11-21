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
// centre and a ghost by the process of the atom of the lower id, which the
// other process sees too, so that exactly one of them computes it; a centre and
// a ghost of itself from the vector whose first component not zero is
// positive. The contributions to the ghosts go back to their atoms.

#pragma once

#include <cstdint>

#include "core/index.h"

namespace mdapy::calculators {

// What decides before the pair vector: false for a pair another row or
// process computes.
inline bool once_first(std::int64_t i, std::int64_t j, const std::int8_t* shift,
                       std::int64_t centers, const std::int64_t* ids) {
    if (j >= centers) {
        return ids[i] <= ids[j];
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

}  // namespace mdapy::calculators

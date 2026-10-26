// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The integer type of atom indices in compiled results such as neighbor
// lists. It has 32 bits, which halves their memory; mdapy built with
// -Ccmake.define.MDAPY_INDEX64=ON uses 64 bits, for more than 2^31 - 1 atoms.

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace mdapy {

#ifdef MDAPY_INDEX64
using index_t = std::int64_t;
#else
using index_t = std::int32_t;
#endif

// Throws std::overflow_error when n atoms do not fit index_t.
inline void check_atom_count(std::int64_t n) {
    if constexpr (sizeof(index_t) < sizeof(std::int64_t)) {
        if (n > std::numeric_limits<index_t>::max()) {
            throw std::overflow_error(
                std::to_string(n) +
                " atoms do not fit 32-bit atom indices; install mdapy built with "
                "-Ccmake.define.MDAPY_INDEX64=ON for 64-bit indices");
        }
    }
}

}  // namespace mdapy

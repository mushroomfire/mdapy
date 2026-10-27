// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The signatures of common neighbor analysis, from the bonds among the
// neighbors of an atom as bitmasks.

#pragma once

#include <algorithm>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace mdapy::identify {

// The number of set bits, by shifts and adds: std::bitset::count calls a
// library function on processors without a popcount instruction in the
// baseline instruction set, which keeps the loops below from being
// vectorized.
inline int count(std::uint32_t bits) {
    bits = bits - ((bits >> 1) & 0x55555555u);
    bits = (bits & 0x33333333u) + ((bits >> 2) & 0x33333333u);
    bits = (bits + (bits >> 4)) & 0x0f0f0f0fu;
    return static_cast<int>((bits * 0x01010101u) >> 24);
}

// The CNA signature of the bond to neighbor j among m neighbors whose
// bonds are given as bitmasks: the number of common neighbors, the number
// of bonds among them and the number of bonds in the largest connected
// group of those bonds.
struct Signature {
    int common, bonds, chain;
};

// The index of the lowest bit set in bits, which is not zero.
inline int lowest(std::uint32_t bits) {
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, bits);
    return static_cast<int>(index);
#else
    return __builtin_ctz(bits);
#endif
}

inline Signature signature(const std::uint32_t* bonds, int j) {
    const std::uint32_t common = bonds[j];
    // the loops run over the neighbors of a set, a handful at most
    int total = 0, most = 0;  // twice the bonds, and the most bonds of a neighbor
    for (std::uint32_t rest = common; rest != 0; rest &= rest - 1) {
        const int degree = count(bonds[lowest(rest)] & common);
        total += degree;
        most = std::max(most, degree);
    }
    // the bonds in the largest connected group of them
    int chain = 0;
    if (total == 0) {
        // no bonds
    } else if (most == 1) {
        chain = 1;  // the bonds share no neighbor
    } else if (total == 4) {
        chain = 2;  // two bonds with a neighbor in common
    } else {
        // connected groups of the common neighbors joined by bonds
        std::uint32_t left = common;
        while (left != 0) {
            std::uint32_t group = left & (~left + 1u);  // lowest remaining neighbor
            std::uint32_t frontier = group;
            while (frontier != 0) {
                std::uint32_t next = 0;
                for (std::uint32_t rest = frontier; rest != 0; rest &= rest - 1) {
                    next |= bonds[lowest(rest)] & common;
                }
                frontier = next & ~group;
                group |= next;
            }
            int inside = 0;
            for (std::uint32_t rest = group; rest != 0; rest &= rest - 1) {
                inside += count(bonds[lowest(rest)] & group);
            }
            chain = std::max(chain, inside / 2);
            left &= ~group;
        }
    }
    return {count(common), total / 2, chain};
}

}  // namespace mdapy::identify

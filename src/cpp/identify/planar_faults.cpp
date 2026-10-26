// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "identify/planar_faults.h"

#include <array>
#include <vector>

#include "identify/ptm.h"

namespace mdapy::identify {
namespace {

constexpr int kStride = 16;  // neighbors given per atom
// The plane of each neighbor of an hcp atom in the order of the PTM
// template: below (-1), in (0) or above (+1) its basal plane.
constexpr std::array<int, 12> kLayer = {0, 0, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1};
constexpr std::array<int, 6> kBasal = {0, 1, 5, 6, 7, 8};
constexpr std::array<int, 6> kOutOfPlane = {2, 3, 4, 9, 10, 11};

}  // namespace

void planar_faults(const std::int8_t* structure, const index_t* neighbors, std::int64_t n,
                   std::int8_t* faults, const Control& control) {
    std::vector<std::int64_t> hcp;
    for (std::int64_t i = 0; i < n; ++i) {
        faults[i] = kNonHcp;
        if (structure[i] == kPtmHcp) {
            hcp.push_back(i);
        }
    }
    const auto count = static_cast<std::int64_t>(hcp.size());
    const auto row = [&](std::int64_t i) { return neighbors + kStride * i; };
    // two hcp atoms are stacked unless an atom is in both their basal planes
    const auto stacked = [&](std::int64_t a, std::int64_t b) {
        for (int i : kBasal) {
            for (int j : kBasal) {
                if (row(a)[i] == row(b)[j]) {
                    return false;
                }
            }
        }
        return true;
    };
    parallel_for(
        count, 0,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t h = begin; h < end; ++h) {
                const std::int64_t a = hcp[static_cast<std::size_t>(h)];
                int basal = 0, above = 0, below = 0, fcc_above = 0, fcc_below = 0;
                for (int k = 0; k < 12; ++k) {
                    const auto b = static_cast<std::int64_t>(row(a)[k]);
                    if (structure[b] == kPtmHcp) {
                        if (kLayer[k] == 0) {
                            ++basal;
                        } else if (stacked(a, b)) {
                            (kLayer[k] > 0 ? above : below) += 1;
                        }
                    } else if (kLayer[k] != 0 && structure[b] == kPtmFcc) {
                        (kLayer[k] > 0 ? fcc_above : fcc_below) += 1;
                    }
                }
                std::int8_t type = kOtherHcp;
                if ((above != 0) != (below != 0)) {
                    type = kIntrinsicFault;
                } else if (basal >= 1 && above == 0 && below == 0 && fcc_above != 0 &&
                           fcc_below != 0) {
                    type = kTwinBoundary;
                } else if (above != 0 && below != 0) {
                    type = kMultilayerFault;
                }
                faults[a] = type;
            }
        },
        control);
    // the second pass reads the types it has already changed: serial
    for (const std::int64_t a : hcp) {
        if (faults[a] == kTwinBoundary || faults[a] == kOtherHcp) {
            int intrinsic = 0, twin = 0;
            for (int k : kBasal) {
                const auto b = static_cast<std::int64_t>(row(a)[k]);
                if (structure[b] == kPtmHcp) {
                    intrinsic += faults[b] == kIntrinsicFault;
                    twin += faults[b] == kTwinBoundary;
                }
            }
            if (intrinsic != 0 && twin == 0) {
                faults[a] = kIntrinsicFault;
            } else if (intrinsic == 0 && twin != 0) {
                faults[a] = kTwinBoundary;
            }
        } else if (faults[a] == kMultilayerFault) {
            for (int k : kOutOfPlane) {
                const auto b = static_cast<std::int64_t>(row(a)[k]);
                if (structure[b] == kPtmHcp && faults[b] == kIntrinsicFault) {
                    faults[b] = kMultilayerFault;
                }
            }
        }
    }
}

}  // namespace mdapy::identify

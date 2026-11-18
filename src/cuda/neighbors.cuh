// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "cuda/device.cuh"

namespace mdapy::cuda {

struct Neighbors::Impl {
    double cutoff = 0.0;
    double skin = 0.0;
    std::int64_t n = 0;
    std::int64_t builds = 0;
    int most = 0;                    // columns of the list
    Buffer<std::int32_t> indices;    // (most, n): neighbor k of atom i at k * n + i
    Buffer<std::int32_t> counts;     // (n,)
    // the state of the last build, to tell when to build again
    Buffer<double4> x_built;
    Box box_built{};
    std::uint64_t atoms_version = 0;
    // the cells
    Buffer<std::int32_t> keys, keys_sorted, order, order_sorted;
    Buffer<std::int32_t> cell_begin, cell_end;
    Buffer<unsigned char> sort_scratch;
    Buffer<double> displacement;     // (n,) for the test of the skin
    Buffer<int> found;               // the most neighbors an atom has
};

}  // namespace mdapy::cuda

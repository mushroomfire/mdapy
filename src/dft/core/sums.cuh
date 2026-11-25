// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Sums over the blocks of a kernel in an order that is always the same: each
// block writes its partial sums, and one block adds them up, block after
// block, so that a sum does not depend on the order in which the blocks end
// (as one by atomicAdd does) and a calculation gives the same numbers each
// time it runs on the same GPU.

#pragma once

#include "dft/core/portable.h"

#include <cstddef>

#include "dft/core/check.h"

namespace mdapy::dft {

constexpr int kSumThreads = 256;

// out[d] = sum over b < count of partial[b width + d], d < Width, by one block
// of kSumThreads.
template <int Width>
__global__ void ordered_sums_kernel(int count, const double* __restrict__ partial, double* __restrict__ out) {
    __shared__ double part[Width][kSumThreads];
    const int t = static_cast<int>(threadIdx.x);
    double local[Width] = {};
    for (int b = t; b < count; b += kSumThreads)
        for (int d = 0; d < Width; ++d) local[d] += partial[static_cast<std::size_t>(b) * Width + d];
    for (int d = 0; d < Width; ++d) part[d][t] = local[d];
    __syncthreads();
    for (int s = kSumThreads / 2; s > 0; s >>= 1) {
        if (t < s)
            for (int d = 0; d < Width; ++d) part[d][t] += part[d][t + s];
        __syncthreads();
    }
    if (t == 0)
        for (int d = 0; d < Width; ++d) out[d] = part[d][0];
}

// The Width sums of the partial sums of count blocks, Width numbers a block in
// partial, into host (Width numbers); partial holds (count + 1) Width numbers,
// the last Width taking the sums on their way.
template <int Width>
void ordered_sums(int count, double* partial, double* host) {
    double* out = partial + static_cast<std::size_t>(count) * Width;
    ordered_sums_kernel<Width><<<1, kSumThreads>>>(count, partial, out);
    check_cuda(cudaGetLastError(), "a sum over the blocks");
    check_cuda(cudaMemcpy(host, out, Width * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy from the GPU");
}

}  // namespace mdapy::dft

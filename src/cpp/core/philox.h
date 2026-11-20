// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

// Counter-based random numbers: Philox4x64-10 of Salmon, Moraes, Dror and Shaw,
// "Parallel random numbers: as easy as 1, 2, 3" (SC11, 2011), the generator of
// numpy.random.Philox. A number depends only on its counter and key, not on
// what was drawn before, so that each atom draws its own noise from its global
// index and the step, whatever thread, device or process computes it, and a run
// continued from a restart draws what it would have drawn. Compiled by the host
// compiler and by nvcc alike, so that the processor and the GPU draw the same.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define MDAPY_HOST_DEVICE __host__ __device__
#else
#define MDAPY_HOST_DEVICE
#endif

namespace mdapy::random {

struct Block {
    std::uint64_t v[4];
};

// The high and low 64 bits of the product of a and b.
MDAPY_HOST_DEVICE inline void multiply(std::uint64_t a, std::uint64_t b, std::uint64_t& high,
                                       std::uint64_t& low) {
#if defined(__CUDA_ARCH__)
    high = __umul64hi(a, b);
    low = a * b;
#elif defined(__SIZEOF_INT128__)
    const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
    high = static_cast<std::uint64_t>(p >> 64);
    low = static_cast<std::uint64_t>(p);
#else
    // four products of 32-bit halves
    const std::uint64_t a0 = a & 0xffffffffu, a1 = a >> 32;
    const std::uint64_t b0 = b & 0xffffffffu, b1 = b >> 32;
    const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const std::uint64_t middle = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    high = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
    low = a * b;
#endif
}

// The ten rounds of Philox4x64 on a counter with a key.
MDAPY_HOST_DEVICE inline Block philox(Block counter, std::uint64_t key0, std::uint64_t key1) {
    constexpr std::uint64_t kMultiply0 = 0xD2E7470EE14C6C93ULL;
    constexpr std::uint64_t kMultiply1 = 0xCA5A826395121157ULL;
    constexpr std::uint64_t kWeyl0 = 0x9E3779B97F4A7C15ULL;
    constexpr std::uint64_t kWeyl1 = 0xBB67AE8584CAA73BULL;
    std::uint64_t* c = counter.v;
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            key0 += kWeyl0;
            key1 += kWeyl1;
        }
        std::uint64_t high0, low0, high1, low1;
        multiply(kMultiply0, c[0], high0, low0);
        multiply(kMultiply1, c[2], high1, low1);
        const std::uint64_t next[4] = {high1 ^ c[1] ^ key0, low1, high0 ^ c[3] ^ key1, low0};
        for (int k = 0; k < 4; ++k) {
            c[k] = next[k];
        }
    }
    return counter;
}

// A number of (0, 1) from the top 53 bits of a word, never 0 or 1.
MDAPY_HOST_DEVICE inline double open_uniform(std::uint64_t word) {
    return (static_cast<double>(word >> 11) + 0.5) * 0x1.0p-53;
}

// Three Gaussian numbers of mean 0 and variance 1, by the method of Box and
// Muller, from one block of the counter (index, draw, 0, 0) with the key
// (seed, stream): the noise of atom index at the draw-th call of a stream.
MDAPY_HOST_DEVICE inline void normal3(std::uint64_t seed, std::uint64_t stream, std::uint64_t index,
                                      std::uint64_t draw, double out[3]) {
    constexpr double kTwoPi = 6.283185307179586476925;
    const Block counter{{index, draw, 0, 0}};
    const Block words = philox(counter, seed, stream);
    const double r0 = std::sqrt(-2.0 * std::log(open_uniform(words.v[0])));
    const double r1 = std::sqrt(-2.0 * std::log(open_uniform(words.v[2])));
    const double a0 = kTwoPi * open_uniform(words.v[1]);
    const double a1 = kTwoPi * open_uniform(words.v[3]);
    out[0] = r0 * std::cos(a0);
    out[1] = r0 * std::sin(a0);
    out[2] = r1 * std::cos(a1);
}

}  // namespace mdapy::random

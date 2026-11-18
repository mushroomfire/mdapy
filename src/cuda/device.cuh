// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// What the kernels share: device memory, errors, the box on the device, the
// minimum image and sums over the atoms.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cuda/cuda.h"

namespace mdapy::cuda {

// Throws std::runtime_error naming the call when a CUDA call fails.
void check(cudaError_t error, const char* call);
#define MDAPY_CUDA(call) ::mdapy::cuda::check((call), #call)
// After a kernel launch.
#define MDAPY_LAUNCHED() ::mdapy::cuda::check(cudaGetLastError(), "kernel launch")

// Device memory of n elements, freed with the buffer.
template <class T>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(std::size_t n) { resize(n); }
    ~Buffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Its contents are lost when the size changes.
    void resize(std::size_t n) {
        if (n == size_) {
            return;
        }
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = nullptr;
        size_ = 0;
        if (n > 0) {
            MDAPY_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T)));
            size_ = n;
        }
    }
    void upload(const T* host, std::size_t n) {
        resize(n);
        if (n > 0) {
            MDAPY_CUDA(cudaMemcpy(data_, host, n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    void download(T* host, std::size_t n) const {
        if (n > 0) {
            MDAPY_CUDA(cudaMemcpy(host, data_, n * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
    void swap(Buffer& other) noexcept {
        T* d = data_;
        data_ = other.data_;
        other.data_ = d;
        const std::size_t s = size_;
        size_ = other.size_;
        other.size_ = s;
    }
    void zero() {
        if (size_ > 0) {
            MDAPY_CUDA(cudaMemset(data_, 0, size_ * sizeof(T)));
        }
    }
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

constexpr int kBlock = 128;

inline unsigned blocks(std::int64_t n) {
    return static_cast<unsigned>((n + kBlock - 1) / kBlock);
}

// Four doubles aligned to 32 bytes, read in one load; CUDA 13 deprecates
// its double4, whose alignment depends on the compiler.
struct alignas(32) double4a {
    double x, y, z, w;
};

// Four numbers of a precision: float4 or double4a.
template <class Real>
struct Four;
template <>
struct Four<float> {
    using type = float4;
};
template <>
struct Four<double> {
    using type = double4a;
};
template <class Real>
using Real4 = typename Four<Real>::type;

// The box in a precision: its vectors as rows, their inverse, the periodic
// directions, and whether it is orthogonal, when the minimum image needs no
// fractional coordinates.
template <class Real>
struct DeviceBox {
    Real h[9];
    Real inverse[9];
    Real lo[3];
    int pbc[3];
    int orthogonal;
};

// The box of the host in a precision; throws std::invalid_argument for one
// of no volume.
template <class Real>
DeviceBox<Real> device_box(const Box& box);

// A neighbor in a list: its index in the low bits, the image of the box it
// is reached through in the high ones, from 0 to 26 (see Neighbors::get).
constexpr int kIndexBits = 27;
constexpr std::uint32_t kIndexMask = (1u << kIndexBits) - 1u;

// The steps of the codes of pairs, h_k / n_k, rows, in a precision.
template <class Real>
struct Images {
    Real h[9];
};

// x_j - x_i of a pair from l_j - l_i: plus d . steps, d of -1, 0 or 1 along
// each vector by its code (Neighbors::Impl). Most pairs have one of a few
// codes; an array of the 27 vectors indexed at run time would be read from
// local memory.
template <class Real>
__device__ inline void add_image(const Images<Real>& b, std::uint32_t code, Real& x, Real& y,
                                 Real& z) {
    if (code == 13u) return;
    const int s0 = static_cast<int>(code % 3u) - 1;
    const int s1 = static_cast<int>((code / 3u) % 3u) - 1;
    const int s2 = static_cast<int>(code / 9u) - 1;
    x += s0 * b.h[0] + s1 * b.h[3] + s2 * b.h[6];
    y += s0 * b.h[1] + s1 * b.h[4] + s2 * b.h[7];
    z += s0 * b.h[2] + s1 * b.h[5] + s2 * b.h[8];
}

// Sums of columns: out[k] = sum over i of data[k * n + i] for k < columns,
// on the host, in an order that does not change from run to run.
void sum_columns(const double* data, int columns, std::int64_t n, double* out);

// The largest of values (n,), NaN if any is NaN.
double largest(const double* values, std::int64_t n);

}  // namespace mdapy::cuda

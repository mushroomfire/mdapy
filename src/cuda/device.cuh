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

// Four numbers of a precision: float4 or double4.
template <class Real>
struct Four;
template <>
struct Four<float> {
    using type = float4;
};
template <>
struct Four<double> {
    using type = double4;
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

// The shortest image of a vector between two atoms along the periodic
// directions; exact for vectors shorter than half the thickness of the box.
template <class Real>
__device__ inline void minimum_image(const DeviceBox<Real>& b, Real& x, Real& y, Real& z) {
    if (b.orthogonal) {
        if (b.pbc[0]) x -= b.h[0] * rint(x * b.inverse[0]);
        if (b.pbc[1]) y -= b.h[4] * rint(y * b.inverse[4]);
        if (b.pbc[2]) z -= b.h[8] * rint(z * b.inverse[8]);
        return;
    }
    // fractional coordinates s of r = s h, rows of h the vectors
    Real s[3];
    for (int k = 0; k < 3; ++k) {
        s[k] = x * b.inverse[k] + y * b.inverse[3 + k] + z * b.inverse[6 + k];
        if (b.pbc[k]) s[k] -= rint(s[k]);
    }
    x = s[0] * b.h[0] + s[1] * b.h[3] + s[2] * b.h[6];
    y = s[0] * b.h[1] + s[1] * b.h[4] + s[2] * b.h[7];
    z = s[0] * b.h[2] + s[1] * b.h[5] + s[2] * b.h[8];
}

// Sums of columns: out[k] = sum over i of data[k * n + i] for k < columns,
// on the host, in an order that does not change from run to run.
void sum_columns(const double* data, int columns, std::int64_t n, double* out);

// The largest of values (n,), NaN if any is NaN.
double largest(const double* values, std::int64_t n);

}  // namespace mdapy::cuda

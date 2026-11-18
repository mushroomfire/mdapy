// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/device.cuh"

namespace mdapy::cuda {

void check(cudaError_t error, const char* call) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error) + " (" +
                                 call + ")");
    }
}

std::string device_name() {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess || count == 0) {
        throw std::runtime_error(
            std::string("no CUDA device: ") +
            (error != cudaSuccess ? cudaGetErrorString(error) : "none was found"));
    }
    int device = 0;
    MDAPY_CUDA(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    MDAPY_CUDA(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}

template <class Real>
DeviceBox<Real> device_box(const Box& box) {
    const double* h = box.h;
    // the inverse by cofactors: inverse[a][k] = cofactor[k][a] / det
    const double c[9] = {
        h[4] * h[8] - h[5] * h[7], h[5] * h[6] - h[3] * h[8], h[3] * h[7] - h[4] * h[6],
        h[2] * h[7] - h[1] * h[8], h[0] * h[8] - h[2] * h[6], h[1] * h[6] - h[0] * h[7],
        h[1] * h[5] - h[2] * h[4], h[2] * h[3] - h[0] * h[5], h[0] * h[4] - h[1] * h[3],
    };
    const double det = h[0] * c[0] + h[1] * c[1] + h[2] * c[2];
    if (!(std::fabs(det) > 0.0) || !std::isfinite(det)) {
        throw std::invalid_argument("the box has no volume");
    }
    DeviceBox<Real> b{};
    bool orthogonal = true;
    for (int a = 0; a < 3; ++a) {
        for (int k = 0; k < 3; ++k) {
            b.h[3 * a + k] = static_cast<Real>(h[3 * a + k]);
            b.inverse[3 * a + k] = static_cast<Real>(c[3 * k + a] / det);
            if (a != k && h[3 * a + k] != 0.0) {
                orthogonal = false;
            }
        }
        b.lo[a] = static_cast<Real>(box.lo[a]);
        b.pbc[a] = box.pbc[a] ? 1 : 0;
    }
    b.orthogonal = orthogonal ? 1 : 0;
    return b;
}

template DeviceBox<float> device_box<float>(const Box&);
template DeviceBox<double> device_box<double>(const Box&);


namespace {

constexpr int kGrid = 256;  // blocks of the first pass of a sum

struct Add {
    __device__ double operator()(double a, double b) const { return a + b; }
    static constexpr double identity = 0.0;
};

struct Largest {
    // NaN wins, so that a position or force gone wrong is seen
    __device__ double operator()(double a, double b) const {
        return (a > b || a != a) ? a : b;
    }
    static constexpr double identity = -INFINITY;
};

// The reduction of a block, in a fixed order.
template <class Op>
__device__ double block_reduce(double value, double* shared) {
    const Op op;
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            shared[threadIdx.x] = op(shared[threadIdx.x], shared[threadIdx.x + half]);
        }
        __syncthreads();
    }
    return shared[0];
}

// First pass: block b of column k reduces the elements b * kBlock + t +
// m * kGrid * kBlock of the column into partial[k * kGrid + b].
template <class Op>
__global__ void reduce_first(const double* data, std::int64_t n, double* partial) {
    __shared__ double shared[kBlock];
    const Op op;
    const double* column = data + static_cast<std::int64_t>(blockIdx.y) * n;
    double value = Op::identity;
    const std::int64_t stride = static_cast<std::int64_t>(kGrid) * kBlock;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kBlock + threadIdx.x; i < n;
         i += stride) {
        value = op(value, column[i]);
    }
    const double total = block_reduce<Op>(value, shared);
    if (threadIdx.x == 0) {
        partial[blockIdx.y * kGrid + blockIdx.x] = total;
    }
}

// Second pass: one block per column reduces its kGrid partial results.
template <class Op>
__global__ void reduce_second(const double* partial, double* out) {
    __shared__ double shared[kBlock];
    const Op op;
    double value = Op::identity;
    for (int b = threadIdx.x; b < kGrid; b += kBlock) {
        value = op(value, partial[blockIdx.x * kGrid + b]);
    }
    const double total = block_reduce<Op>(value, shared);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = total;
    }
}

std::mutex scratch_mutex;

template <class Op>
void reduce(const double* data, int columns, std::int64_t n, double* out) {
    const std::lock_guard<std::mutex> lock(scratch_mutex);
    static Buffer<double> partial;
    static Buffer<double> result;
    partial.resize(static_cast<std::size_t>(columns) * kGrid);
    if (result.size() < static_cast<std::size_t>(columns)) {
        result.resize(static_cast<std::size_t>(columns));
    }
    reduce_first<Op><<<dim3(kGrid, columns), kBlock>>>(data, n, partial.data());
    MDAPY_LAUNCHED();
    reduce_second<Op><<<columns, kBlock>>>(partial.data(), result.data());
    MDAPY_LAUNCHED();
    result.download(out, static_cast<std::size_t>(columns));
}

}  // namespace

void sum_columns(const double* data, int columns, std::int64_t n, double* out) {
    reduce<Add>(data, columns, n, out);
}

double largest(const double* values, std::int64_t n) {
    double out = -INFINITY;
    reduce<Largest>(values, 1, n, &out);
    return out;
}

}  // namespace mdapy::cuda

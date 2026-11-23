// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Memory of the GPU owned by an object: allocated once, freed with it, moved
// but never copied. The complex numbers of the device are cuDoubleComplex, the
// layout of std::complex<double>. The memory is that of the stream-ordered pool
// on the default stream of the calling thread, which cudaFree, waiting for the
// whole device, would not let runs on several threads overlap; the pool keeps
// what it is given back.

#pragma once

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "dft/core/check.h"

namespace mdapy::dft {

using Complex = cuDoubleComplex;

// the pool of the current device keeps the memory given back to it, once
inline void keep_pool() {
    static const bool kept = [] {
        int device = 0;
        cudaMemPool_t pool{};
        if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess)
            return false;
        unsigned long long most = ~0ULL;
        return cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &most) == cudaSuccess;
    }();
    (void)kept;
}

template <class T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t size) { resize(size); }
    ~DeviceArray() { release(); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    // Holds size elements: memory of another size is freed and allocated
    // again, its values lost; memory of this size is kept as it is.
    void resize(std::size_t size) {
        if (size == size_) return;
        release();
        if (size > 0) {
            void* memory = nullptr;
            keep_pool();
            check_cuda(cudaMallocAsync(&memory, size * sizeof(T), cudaStreamPerThread), "cudaMallocAsync");
            data_ = static_cast<T*>(memory);
            size_ = size;
        }
    }

    // Holds at least size elements, keeping the memory and its values when it
    // already does.
    void reserve(std::size_t size) {
        if (size > size_) resize(size);
    }

    // The first count elements from the host, growing the memory if needed.
    void assign(const T* host, std::size_t count) {
        reserve(count);
        if (count > 0)
            check_cuda(cudaMemcpy(data_, host, count * sizeof(T), cudaMemcpyHostToDevice),
                       "cudaMemcpy to the GPU");
    }

    void zero() {
        if (size_ > 0) check_cuda(cudaMemset(data_, 0, size_ * sizeof(T)), "cudaMemset");
    }

    void upload(const T* host, std::size_t count) {
        resize(count);
        if (count > 0)
            check_cuda(cudaMemcpy(data_, host, count * sizeof(T), cudaMemcpyHostToDevice),
                       "cudaMemcpy to the GPU");
    }

    void download(T* host, std::size_t count) const {
        if (count > size_) throw std::out_of_range("a download beyond the array of the GPU");
        if (count > 0)
            check_cuda(cudaMemcpy(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                       "cudaMemcpy from the GPU");
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    void release() noexcept {
        if (data_ != nullptr) cudaFreeAsync(data_, cudaStreamPerThread);
        data_ = nullptr;
        size_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace mdapy::dft

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
// what it is given back, for the next arrays, until give_back returns it to
// the device (at the end of each run of the SCF).

#pragma once

#include "dft/core/portable.h"

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "dft/core/check.h"

namespace mdapy::dft {

using Complex = cuDoubleComplex;

// the pool of the current device keeps the memory given back to it, set once
// for each device (a bit each of the first 64)
inline void keep_pool() {
    static std::atomic<unsigned long long> kept{0};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return;
    const unsigned long long bit = 1ULL << device;
    if ((kept.load(std::memory_order_relaxed) & bit) != 0) return;
    cudaMemPool_t pool{};
    unsigned long long most = ~0ULL;
    if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess &&
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &most) == cudaSuccess)
        kept.fetch_or(bit, std::memory_order_relaxed);
}

// The memory of the stream-ordered pool of the current device, bytes: in use
// by the arrays of the whole process and reserved by the pool from the
// device, now and at their largest since reset_pool_peaks; and the free and
// total memory of the device, cudaMemGetInfo's.
struct PoolMemory {
    std::size_t in_use = 0, in_use_peak = 0, reserved = 0, reserved_peak = 0;
    std::size_t free = 0, total = 0;
};

inline cudaMemPool_t current_pool() {
    int device = 0;
    cudaMemPool_t pool{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaDeviceGetDefaultMemPool(&pool, device), "cudaDeviceGetDefaultMemPool");
    return pool;
}

inline PoolMemory pool_memory() {
    const cudaMemPool_t pool = current_pool();
    auto get = [&](cudaMemPoolAttr attribute) {
        unsigned long long value = 0;
        check_cuda(cudaMemPoolGetAttribute(pool, attribute, &value), "cudaMemPoolGetAttribute");
        return static_cast<std::size_t>(value);
    };
    PoolMemory m;
    m.in_use = get(cudaMemPoolAttrUsedMemCurrent);
    m.in_use_peak = get(cudaMemPoolAttrUsedMemHigh);
    m.reserved = get(cudaMemPoolAttrReservedMemCurrent);
    m.reserved_peak = get(cudaMemPoolAttrReservedMemHigh);
    check_cuda(cudaMemGetInfo(&m.free, &m.total), "cudaMemGetInfo");
    return m;
}

// the peaks of pool_memory set to the present values
inline void reset_pool_peaks() {
    const cudaMemPool_t pool = current_pool();
    unsigned long long zero = 0;
    check_cuda(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero), "cudaMemPoolSetAttribute");
    check_cuda(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &zero), "cudaMemPoolSetAttribute");
}

// The memory of the pool beyond keep bytes that no array holds given back to
// the device, once what this thread freed is freed on the GPU; errors are
// ignored, the memory then staying with the pool.
inline void give_back(std::size_t keep) noexcept {
    int device = 0;
    cudaMemPool_t pool{};
    if (cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess || cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess)
        return;
    static_cast<void>(cudaMemPoolTrimTo(pool, keep));
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

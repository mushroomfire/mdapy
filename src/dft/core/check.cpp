// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/core/check.h"

#include <new>
#include <stdexcept>
#include <string>

#ifdef MDAPY_DFT_PROFILE
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <utility>

#include "dft/core/profile.h"
#endif

namespace mdapy::dft {

void check_cuda(cudaError_t status, const char* call) {
    if (status == cudaSuccess) return;
    if (status == cudaErrorMemoryAllocation) {
        static_cast<void>(cudaGetLastError());  // the error is not sticky: clear it
        throw std::bad_alloc();
    }
    throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(status));
}

void check_cufft(cufftResult status, const char* call) {
    if (status == CUFFT_SUCCESS) return;
    if (status == CUFFT_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(std::string(call) + " failed with cuFFT status " +
                             std::to_string(static_cast<int>(status)));
}

const char* device_name() {
    static cudaDeviceProp properties{};
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    return properties.name;
}

int device_count() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) return 0;
    check_cuda(status, "cudaGetDeviceCount");
    return count;
}

}  // namespace mdapy::dft

#ifdef MDAPY_DFT_PROFILE
namespace mdapy::dft {

namespace {

struct Rows {
    std::mutex lock;
    std::map<std::string, std::pair<long long, double>> rows;
    void print(const char* label) {
        const std::lock_guard<std::mutex> guard(lock);
        for (const auto& [path, value] : rows)
            std::fprintf(stderr, "SPAN %s %s %lld %.6f\n", label, path.c_str(), value.first, value.second);
        rows.clear();
    }
    ~Rows() { print("end"); }
};

Rows& rows() {
    static Rows r;
    return r;
}

thread_local std::string path;

long long now() {
    cudaStreamSynchronize(cudaStreamPerThread);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

Span::Span(const char* name) : length_(path.size()) {
    rows();  // made before the first span ends
    path += '/';
    path += name;
    start_ = now();
}

Span::~Span() {
    const double seconds = 1e-9 * static_cast<double>(now() - start_);
    Rows& r = rows();
    {
        const std::lock_guard<std::mutex> guard(r.lock);
        auto& row = r.rows[path];
        ++row.first;
        row.second += seconds;
    }
    path.resize(length_);
}

}  // namespace mdapy::dft

extern "C" __attribute__((visibility("default"))) void mdapy_dft_profile(const char* label) {
    mdapy::dft::rows().print(label);
}
#endif

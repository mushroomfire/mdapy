// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/core/check.h"

#include <new>
#include <stdexcept>
#include <string>

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

// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The status of the calls to CUDA and its libraries, turned into exceptions that
// name the call: std::bad_alloc when the GPU is out of memory, std::runtime_error
// otherwise.

#pragma once

#include <cuda_runtime_api.h>
#include <cufft.h>

namespace mdapy::dft {

void check_cuda(cudaError_t status, const char* call);
void check_cufft(cufftResult status, const char* call);

// The name of the GPU in use and the number of GPUs; throw without one.
const char* device_name();
int device_count();

}  // namespace mdapy::dft

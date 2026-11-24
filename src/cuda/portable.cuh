// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The runtime of the device: CUDA for NVIDIA GPUs, or HIP for AMD GPUs, for
// which the sources are compiled as they are, their CUDA names those of HIP
// by the aliases below. Only what the kernels use is named here.
//
// Lanes: the kernels that share work among the threads of a warp do it among
// 32 lanes, threads 32 w to 32 w + 31 of a block, whatever the device runs
// together: a warp of 32 on NVIDIA GPUs, half a wavefront of 64 on AMD GPUs,
// where two such groups share a wavefront. Blocks are whole multiples of 64.

#pragma once

// __HIP__ where the clang of ROCm compiles kernels, __HIP_PLATFORM_AMD__ where
// the host compiler compiles what holds memory of the device (the DFT)
#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

#define MDAPY_GPU_API "HIP"
#define MDAPY_GPU_HIP 1

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
// freed in destructors, where nothing can be done about an error
#define cudaFree(pointer) static_cast<void>(hipFree(pointer))
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaFreeHost(pointer) static_cast<void>(hipHostFree(pointer))
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t

// The value of the lane offset higher among the lanes of its group.
#define MDAPY_SHUFFLE_DOWN(value, offset) __shfl_down((value), (offset), ::mdapy::cuda::kLanes)
// What the lanes of a group wrote to shared memory is there for the others:
// HIP has no barrier of a wavefront, so that of the block.
#define MDAPY_SYNC_LANES() __syncthreads()

#else

#include <cuda_runtime.h>

#define MDAPY_GPU_API "CUDA"

#define MDAPY_SHUFFLE_DOWN(value, offset) \
    __shfl_down_sync(0xffffffffu, (value), (offset), ::mdapy::cuda::kLanes)
#define MDAPY_SYNC_LANES() __syncwarp()

#endif

namespace mdapy::cuda {

// The threads of a group that shares work by shuffles; see above.
constexpr int kLanes = 32;

}  // namespace mdapy::cuda

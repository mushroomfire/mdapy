// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The runtime and the libraries of the device for the DFT: CUDA with cuBLAS,
// cuFFT and cuSOLVER for NVIDIA GPUs, or HIP with hipBLAS, hipFFT and hipSOLVER
// (on rocBLAS, rocFFT and rocSOLVER) for AMD GPUs, for which the sources are
// compiled as they are, their CUDA names those of HIP by the aliases of
// cuda/portable.cuh and those below. Only what the DFT uses is named here. The
// libraries of HIP take the arguments of those of CUDA, in the same order.
//
// Streams: with CUDA the kernels, the copies and the libraries of a host
// thread run on its own default stream (the build makes cudaStreamPerThread
// the stream of what names none), so that runs on several threads overlap.
// HIP has no such default: what names no stream runs on the one default
// stream of the device, so with HIP cudaStreamPerThread is that stream, and
// the runs of several threads take turns on the device, in order.

#pragma once

#include "cuda/portable.cuh"

#if defined(MDAPY_GPU_HIP)

#include <hip/hip_complex.h>
#include <hipblas/hipblas.h>
#include <hipfft/hipfft.h>
#include <hipsolver/hipsolver.h>

// the runtime
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaStreamPerThread (static_cast<hipStream_t>(nullptr))
#define cudaMallocAsync hipMallocAsync
// freed in destructors, where nothing can be done about an error
#define cudaFreeAsync(pointer, stream) static_cast<void>(hipFreeAsync((pointer), (stream)))
#define cudaMemPool_t hipMemPool_t
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#define cudaMemPoolAttr hipMemPoolAttr
#define cudaMemPoolAttrUsedMemCurrent hipMemPoolAttrUsedMemCurrent
#define cudaMemPoolAttrUsedMemHigh hipMemPoolAttrUsedMemHigh
#define cudaMemPoolAttrReservedMemCurrent hipMemPoolAttrReservedMemCurrent
#define cudaMemPoolAttrReservedMemHigh hipMemPoolAttrReservedMemHigh
#define cudaMemPoolTrimTo hipMemPoolTrimTo
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemset2D hipMemset2D
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

// complex numbers: x and y, the layout of std::complex
#define cuComplex hipFloatComplex
#define cuDoubleComplex hipDoubleComplex
#define make_cuComplex make_hipFloatComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define cuConjf hipConjf
#define cuConj hipConj

// hipBLAS, with the complex numbers of HIP (HIPBLAS_V2, set by the build)
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasOperation_t hipblasOperation_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasScopy hipblasScopy
#define cublasDcopy hipblasDcopy
#define cublasCcopy hipblasCcopy
#define cublasZcopy hipblasZcopy
#define cublasSscal hipblasSscal
#define cublasDscal hipblasDscal
#define cublasSger hipblasSger
#define cublasDger hipblasDger
#define cublasSgemm hipblasSgemm
#define cublasDgemm hipblasDgemm
#define cublasCgemm hipblasCgemm
#define cublasZgemm hipblasZgemm
#define cublasDgemmStridedBatched hipblasDgemmStridedBatched
#define cublasStrsm hipblasStrsm
#define cublasDtrsm hipblasDtrsm
#define cublasCtrsm hipblasCtrsm
#define cublasZtrsm hipblasZtrsm

// hipFFT
#define cufftHandle hipfftHandle
#define cufftResult hipfftResult
#define cufftType hipfftType
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_C2R HIPFFT_C2R
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_D2Z HIPFFT_D2Z
#define CUFFT_Z2D HIPFFT_Z2D
#define CUFFT_Z2Z HIPFFT_Z2Z
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define cufftPlanMany hipfftPlanMany
#define cufftSetStream hipfftSetStream
#define cufftDestroy hipfftDestroy
#define cufftExecR2C hipfftExecR2C
#define cufftExecC2R hipfftExecC2R
#define cufftExecC2C hipfftExecC2C
#define cufftExecD2Z hipfftExecD2Z
#define cufftExecZ2D hipfftExecZ2D
#define cufftExecZ2Z hipfftExecZ2Z

// hipSOLVER, the dense solvers with the arguments of cuSOLVER
#define cusolverDnHandle_t hipsolverDnHandle_t
#define cusolverStatus_t hipsolverStatus_t
#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_ALLOC_FAILED HIPSOLVER_STATUS_ALLOC_FAILED
#define CUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnSetStream hipsolverDnSetStream
#define cusolverDnSsyevd_bufferSize hipsolverDnSsyevd_bufferSize
#define cusolverDnDsyevd_bufferSize hipsolverDnDsyevd_bufferSize
#define cusolverDnCheevd_bufferSize hipsolverDnCheevd_bufferSize
#define cusolverDnZheevd_bufferSize hipsolverDnZheevd_bufferSize
#define cusolverDnSsyevd hipsolverDnSsyevd
#define cusolverDnDsyevd hipsolverDnDsyevd
#define cusolverDnCheevd hipsolverDnCheevd
#define cusolverDnZheevd hipsolverDnZheevd
#define cusolverDnSpotrf_bufferSize hipsolverDnSpotrf_bufferSize
#define cusolverDnDpotrf_bufferSize hipsolverDnDpotrf_bufferSize
#define cusolverDnCpotrf_bufferSize hipsolverDnCpotrf_bufferSize
#define cusolverDnZpotrf_bufferSize hipsolverDnZpotrf_bufferSize
#define cusolverDnSpotrf hipsolverDnSpotrf
#define cusolverDnDpotrf hipsolverDnDpotrf
#define cusolverDnCpotrf hipsolverDnCpotrf
#define cusolverDnZpotrf hipsolverDnZpotrf

#else

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverDn.h>

#endif

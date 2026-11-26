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
// With CUDA the three libraries are not linked: their functions are those of
// the tables of src/dft/core/vendor.h, loaded at their first call.
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

#include "dft/core/vendor.h"

// cuBLAS, cuFFT and cuSOLVER through the tables of src/dft/core/vendor.h,
// loaded at their first call: the names of their headers (cuBLAS's are
// macros for the functions _v2) become those of the tables
#undef cublasCreate
#undef cublasDestroy
#undef cublasSetStream
#undef cublasScopy
#undef cublasDcopy
#undef cublasCcopy
#undef cublasZcopy
#undef cublasSscal
#undef cublasDscal
#undef cublasSger
#undef cublasDger
#undef cublasSgemm
#undef cublasDgemm
#undef cublasCgemm
#undef cublasZgemm
#undef cublasStrsm
#undef cublasDtrsm
#undef cublasCtrsm
#undef cublasZtrsm
#define cublasCreate ::mdapy::dft::vendor::cublas().create
#define cublasDestroy ::mdapy::dft::vendor::cublas().destroy
#define cublasSetStream ::mdapy::dft::vendor::cublas().set_stream
#define cublasScopy ::mdapy::dft::vendor::cublas().scopy
#define cublasDcopy ::mdapy::dft::vendor::cublas().dcopy
#define cublasCcopy ::mdapy::dft::vendor::cublas().ccopy
#define cublasZcopy ::mdapy::dft::vendor::cublas().zcopy
#define cublasSscal ::mdapy::dft::vendor::cublas().sscal
#define cublasDscal ::mdapy::dft::vendor::cublas().dscal
#define cublasSger ::mdapy::dft::vendor::cublas().sger
#define cublasDger ::mdapy::dft::vendor::cublas().dger
#define cublasSgemm ::mdapy::dft::vendor::cublas().sgemm
#define cublasDgemm ::mdapy::dft::vendor::cublas().dgemm
#define cublasCgemm ::mdapy::dft::vendor::cublas().cgemm
#define cublasZgemm ::mdapy::dft::vendor::cublas().zgemm
#define cublasDgemmStridedBatched ::mdapy::dft::vendor::cublas().dgemm_strided_batched
#define cublasStrsm ::mdapy::dft::vendor::cublas().strsm
#define cublasDtrsm ::mdapy::dft::vendor::cublas().dtrsm
#define cublasCtrsm ::mdapy::dft::vendor::cublas().ctrsm
#define cublasZtrsm ::mdapy::dft::vendor::cublas().ztrsm
#define cufftPlanMany ::mdapy::dft::vendor::cufft().plan_many
#define cufftSetStream ::mdapy::dft::vendor::cufft().set_stream
#define cufftDestroy ::mdapy::dft::vendor::cufft().destroy
#define cufftExecR2C ::mdapy::dft::vendor::cufft().exec_r2c
#define cufftExecC2R ::mdapy::dft::vendor::cufft().exec_c2r
#define cufftExecC2C ::mdapy::dft::vendor::cufft().exec_c2c
#define cufftExecD2Z ::mdapy::dft::vendor::cufft().exec_d2z
#define cufftExecZ2D ::mdapy::dft::vendor::cufft().exec_z2d
#define cufftExecZ2Z ::mdapy::dft::vendor::cufft().exec_z2z
#define cusolverDnCreate ::mdapy::dft::vendor::cusolver().create
#define cusolverDnDestroy ::mdapy::dft::vendor::cusolver().destroy
#define cusolverDnSetStream ::mdapy::dft::vendor::cusolver().set_stream
#define cusolverDnSsyevd_bufferSize ::mdapy::dft::vendor::cusolver().ssyevd_buffer_size
#define cusolverDnDsyevd_bufferSize ::mdapy::dft::vendor::cusolver().dsyevd_buffer_size
#define cusolverDnCheevd_bufferSize ::mdapy::dft::vendor::cusolver().cheevd_buffer_size
#define cusolverDnZheevd_bufferSize ::mdapy::dft::vendor::cusolver().zheevd_buffer_size
#define cusolverDnSsyevd ::mdapy::dft::vendor::cusolver().ssyevd
#define cusolverDnDsyevd ::mdapy::dft::vendor::cusolver().dsyevd
#define cusolverDnCheevd ::mdapy::dft::vendor::cusolver().cheevd
#define cusolverDnZheevd ::mdapy::dft::vendor::cusolver().zheevd
#define cusolverDnSpotrf_bufferSize ::mdapy::dft::vendor::cusolver().spotrf_buffer_size
#define cusolverDnDpotrf_bufferSize ::mdapy::dft::vendor::cusolver().dpotrf_buffer_size
#define cusolverDnCpotrf_bufferSize ::mdapy::dft::vendor::cusolver().cpotrf_buffer_size
#define cusolverDnZpotrf_bufferSize ::mdapy::dft::vendor::cusolver().zpotrf_buffer_size
#define cusolverDnSpotrf ::mdapy::dft::vendor::cusolver().spotrf
#define cusolverDnDpotrf ::mdapy::dft::vendor::cusolver().dpotrf
#define cusolverDnCpotrf ::mdapy::dft::vendor::cusolver().cpotrf
#define cusolverDnZpotrf ::mdapy::dft::vendor::cusolver().zpotrf

#endif

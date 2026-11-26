# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# The GPU side for AMD GPUs, the module _hip: the sources of the module _cuda
# (cuda.cmake), compiled as they are with HIP of ROCm, their CUDA names those
# of HIP by src/cuda/portable.cuh. Off by default: it needs ROCm (HIP, and the
# headers of hipCUB, rocPRIM and rocThrust). The kernels are compiled by the
# clang of ROCm as HIP, the host code they share with mdapy._core and the
# bindings by the host compiler. The runtime of HIP is a shared library of
# ROCm, which the machine must have. Its random numbers are those of
# core/philox.h, the processor's.
# It is installed in the package MDAPY_HIP_PACKAGE: mdapy, or mdapy_rocm6 for
# the package mdapy-rocm6.
if(CMAKE_VERSION VERSION_LESS 3.21)
  message(FATAL_ERROR "The GPU side with HIP needs CMake 3.21 or newer")
endif()
# CMAKE_HIP_ARCHITECTURES names the GPUs it is compiled for, as gfx90a for the
# MI200 series; without it, the GPU of the build machine.
enable_language(HIP)
find_package(hip REQUIRED)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
# hipCUB, rocPRIM and rocThrust are headers; their CMake targets would compile
# every source that links them as HIP, the host code too
find_path(MDAPY_HIPCUB_INCLUDE hipcub/hipcub.hpp HINTS "${hip_INCLUDE_DIR}" ENV ROCM_PATH
  PATH_SUFFIXES include REQUIRED)
find_path(MDAPY_ROCPRIM_INCLUDE rocprim/rocprim.hpp HINTS "${hip_INCLUDE_DIR}" ENV ROCM_PATH
  PATH_SUFFIXES include REQUIRED)
find_path(MDAPY_ROCTHRUST_INCLUDE thrust/iterator/counting_iterator.h
  HINTS "${hip_INCLUDE_DIR}" ENV ROCM_PATH PATH_SUFFIXES include REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/gpu_sources.cmake")
set_source_files_properties(${MDAPY_GPU_KERNELS} PROPERTIES LANGUAGE HIP)
add_library(mdapy_hip_kernels STATIC ${MDAPY_GPU_KERNELS} ${MDAPY_GPU_HOST_SOURCES})
set_target_properties(mdapy_hip_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON
  HIP_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
target_include_directories(mdapy_hip_kernels PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(mdapy_hip_kernels SYSTEM PRIVATE
  "${MDAPY_HIPCUB_INCLUDE}" "${MDAPY_ROCPRIM_INCLUDE}" "${MDAPY_ROCTHRUST_INCLUDE}")
target_link_libraries(mdapy_hip_kernels PUBLIC hip::host Threads::Threads)
target_compile_options(mdapy_hip_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off>)
if(MDAPY_INDEX64)
  target_compile_definitions(mdapy_hip_kernels PUBLIC MDAPY_INDEX64)
endif()
nanobind_add_module(_hip STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/bindings.cpp")
target_compile_definitions(_hip PRIVATE MDAPY_GPU_MODULE=_hip)
target_link_libraries(_hip PRIVATE mdapy_hip_kernels)
target_compile_options(_hip PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
if(MDAPY_WERROR)
  target_compile_options(_hip PRIVATE -Werror)
endif()
install(TARGETS _hip LIBRARY DESTINATION ${MDAPY_HIP_PACKAGE})

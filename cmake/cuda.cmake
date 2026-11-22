# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# The GPU side, the module _cuda: atoms, neighbor lists, potentials and the steps of
# molecular dynamics on an NVIDIA GPU. Off by default: it needs the CUDA
# toolkit. The kernels and the host code they share with mdapy._core form a
# static library compiled by nvcc and the host compiler; the bindings are
# compiled by the host compiler alone, so nvcc never reads nanobind. The CUDA
# runtime is linked statically: the module needs only the driver. Its random
# numbers are those of core/philox.h, the processor's.
# It is installed in the package MDAPY_CUDA_PACKAGE: mdapy, or mdapy_cuda12 for
# the package mdapy-cuda12.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
file(GLOB MDAPY_CUDA_KERNELS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
add_library(mdapy_cuda_kernels STATIC ${MDAPY_CUDA_KERNELS}
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/eam.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/pair.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/adp.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/stillinger_weber.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/many_body.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/tersoff.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/meam.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/calculators/nep_model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/parallel.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core/hardware.cpp")
set_target_properties(mdapy_cuda_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
target_include_directories(mdapy_cuda_kernels PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(mdapy_cuda_kernels PUBLIC CUDA::cudart_static Threads::Threads)
# the V100 (sm_70) is built for while CUDA 12 builds it, without its warning
target_compile_options(mdapy_cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off>)
if(MDAPY_INDEX64)
  target_compile_definitions(mdapy_cuda_kernels PUBLIC MDAPY_INDEX64)
endif()
# the lines of the sources in the kernels, for profilers and compute-sanitizer
if(MDAPY_CUDA_LINEINFO)
  target_compile_options(mdapy_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
nanobind_add_module(_cuda STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/bindings.cpp")
target_link_libraries(_cuda PRIVATE mdapy_cuda_kernels)
target_compile_options(_cuda PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
if(MDAPY_WERROR)
  target_compile_options(_cuda PRIVATE -Werror)
endif()
install(TARGETS _cuda LIBRARY DESTINATION ${MDAPY_CUDA_PACKAGE})

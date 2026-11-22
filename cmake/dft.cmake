# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# The plane-wave DFT of MDAPY on the GPU, the module _dft: Kohn-Sham density
# functional theory of periodic cells with cuFFT, cuBLAS and cuSOLVER, in double
# precision. The kernels and the host code form a static library compiled by
# nvcc and the host compiler; the bindings are compiled by the host compiler
# alone, so nvcc never reads nanobind. It links the CUDA runtime and the
# libraries as shared libraries. It is installed in the package
# MDAPY_DFT_PACKAGE, mdapy_dft_cuda12.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
file(GLOB_RECURSE MDAPY_DFT_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/*.cpp")
list(REMOVE_ITEM MDAPY_DFT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/bindings.cpp")
add_library(mdapy_dft STATIC ${MDAPY_DFT_SOURCES})
set_target_properties(mdapy_dft PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
target_include_directories(mdapy_dft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(mdapy_dft PUBLIC CUDA::cudart CUDA::cufft CUDA::cublas CUDA::cusolver)
target_compile_options(mdapy_dft PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off>)
if(MDAPY_CUDA_LINEINFO)
  target_compile_options(mdapy_dft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
nanobind_add_module(_dft STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/bindings.cpp")
target_link_libraries(_dft PRIVATE mdapy_dft)
target_compile_options(_dft PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
if(MDAPY_WERROR)
  target_compile_options(_dft PRIVATE -Werror)
  target_compile_options(mdapy_dft PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
install(TARGETS _dft LIBRARY DESTINATION ${MDAPY_DFT_PACKAGE})

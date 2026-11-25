# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# The plane-wave DFT of MDAPY on the GPU, the module _dft: Kohn-Sham density
# functional theory of periodic cells, beside the GPU side of MDAPY (the module
# _cuda or _hip) and built with it. The kernels and the host code form a
# static library compiled by nvcc (or the clang of ROCm) and the host compiler;
# the bindings are compiled by the host compiler alone, so nvcc never reads
# nanobind. It is installed in the package MDAPY_DFT_PACKAGE: mdapy_cuda12 or
# mdapy_rocm6 for the packages mdapy-cuda12 and mdapy-rocm6, or mdapy.
#
# MDAPY_DFT_API CUDA: with cuFFT, cuBLAS and cuSOLVER. The CUDA runtime is
# linked statically, as in _cuda; the libraries are shared, found when the
# module loads in the folders nvidia/<library>/lib that NVIDIA's packages
# nvidia-cublas-cu12, nvidia-cufft-cu12 and nvidia-cusolver-cu12 install beside
# the package (the extra dft of MDAPY), else where the system has them. The
# module names cuBLASLt and what cuSOLVER needs, cuSPARSE and nvJitLink,
# itself, so that the loader takes them from those folders too: NVIDIA's
# libraries look for one another in their own folder only. The wheel holds
# none of them.
#
# MDAPY_DFT_API HIP: the same sources compiled as they are with HIP of ROCm for
# AMD GPUs, their CUDA names those of HIP by src/dft/core/portable.h, with
# hipFFT, hipBLAS and hipSOLVER, shared libraries of ROCm that the machine must
# have, as the HIP runtime of _hip. CMAKE_HIP_ARCHITECTURES names the GPUs, as
# gfx90a for the MI200 series.
file(GLOB_RECURSE MDAPY_DFT_KERNELS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/*.cu")
file(GLOB_RECURSE MDAPY_DFT_HOST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/*.cpp")
list(REMOVE_ITEM MDAPY_DFT_HOST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/bindings.cpp")
if(MDAPY_DFT_API STREQUAL "HIP")
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(FATAL_ERROR "The DFT with HIP needs CMake 3.21 or newer")
  endif()
  enable_language(HIP)
  find_package(hip REQUIRED)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  # the libraries by their files: their CMake targets would compile every
  # source that links them as HIP, the host code too
  foreach(library hipblas hipfft hipsolver)
    find_library(MDAPY_${library}_LIBRARY ${library} HINTS "${hip_LIB_INSTALL_DIR}" ENV ROCM_PATH
      PATH_SUFFIXES lib REQUIRED)
    find_path(MDAPY_${library}_INCLUDE ${library}/${library}.h HINTS "${hip_INCLUDE_DIR}" ENV ROCM_PATH
      PATH_SUFFIXES include REQUIRED)
  endforeach()
  set_source_files_properties(${MDAPY_DFT_KERNELS} PROPERTIES LANGUAGE HIP)
elseif(MDAPY_DFT_API STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # cuSPARSE and nvJitLink of the toolkit, by the file of their major version
  # too, which their runtime packages alone hold
  foreach(library cusparse nvJitLink)
    find_library(MDAPY_${library}_LIBRARY NAMES ${library} lib${library}.so.12 NAMES_PER_DIR
      HINTS "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
  endforeach()
else()
  message(FATAL_ERROR "MDAPY_DFT_API must be CUDA or HIP, not '${MDAPY_DFT_API}'")
endif()
add_library(mdapy_dft STATIC ${MDAPY_DFT_KERNELS} ${MDAPY_DFT_HOST_SOURCES})
set_target_properties(mdapy_dft PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_include_directories(mdapy_dft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(mdapy_dft PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off>)
if(MDAPY_DFT_API STREQUAL "HIP")
  set_target_properties(mdapy_dft PROPERTIES HIP_VISIBILITY_PRESET hidden)
  target_include_directories(mdapy_dft SYSTEM PUBLIC
    "${MDAPY_hipblas_INCLUDE}" "${MDAPY_hipfft_INCLUDE}" "${MDAPY_hipsolver_INCLUDE}")
  target_link_libraries(mdapy_dft PUBLIC hip::host
    "${MDAPY_hipfft_LIBRARY}" "${MDAPY_hipblas_LIBRARY}" "${MDAPY_hipsolver_LIBRARY}")
  # hipBLAS with the complex numbers of HIP, those of hipFFT and hipSOLVER
  target_compile_definitions(mdapy_dft PUBLIC HIPBLAS_V2)
else()
  set_target_properties(mdapy_dft PROPERTIES CUDA_VISIBILITY_PRESET hidden)
  target_link_libraries(mdapy_dft PUBLIC CUDA::cudart_static CUDA::cufft CUDA::cublas
    CUDA::cublasLt CUDA::cusolver "${MDAPY_cusparse_LIBRARY}" "${MDAPY_nvJitLink_LIBRARY}" ${CMAKE_DL_LIBS})
  # the default stream of each host thread, so that runs on several threads (as
  # the SCFs of several configurations) overlap on the GPU instead of taking turns
  target_compile_definitions(mdapy_dft PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
  target_compile_options(mdapy_dft PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets --default-stream per-thread>)
  if(MDAPY_CUDA_LINEINFO)
    target_compile_options(mdapy_dft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
endif()
nanobind_add_module(_dft STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/bindings.cpp")
target_link_libraries(_dft PRIVATE mdapy_dft)
target_compile_options(_dft PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
if(MDAPY_DFT_API STREQUAL "CUDA")
  # every library named in the module, even those whose functions it does not
  # call (cuBLASLt, cuSPARSE, nvJitLink); the folders of NVIDIA's packages as
  # its RUNPATH, before the system's and after LD_LIBRARY_PATH
  set(MDAPY_NVIDIA "$ORIGIN/../nvidia")
  set_target_properties(_dft PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH
    "${MDAPY_NVIDIA}/cublas/lib:${MDAPY_NVIDIA}/cufft/lib:${MDAPY_NVIDIA}/cusolver/lib:${MDAPY_NVIDIA}/cusparse/lib:${MDAPY_NVIDIA}/nvjitlink/lib")
  target_link_options(_dft PRIVATE "LINKER:--no-as-needed" "LINKER:--enable-new-dtags")
endif()
if(MDAPY_WERROR)
  target_compile_options(_dft PRIVATE -Werror)
  target_compile_options(mdapy_dft PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
install(TARGETS _dft LIBRARY DESTINATION ${MDAPY_DFT_PACKAGE})

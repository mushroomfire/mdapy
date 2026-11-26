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
# MDAPY_DFT_API CUDA: the CUDA runtime linked statically, as in _cuda, and
# NVIDIA's cuFFT, cuBLAS and cuSOLVER loaded at their first use, which MDAPY's
# kernels do without (src/dft/core/vendor.h): the module names none of them,
# and loads where they are missing. They are found in the folders
# nvidia/<library>/lib that NVIDIA's packages nvidia-cublas-cu12,
# nvidia-cufft-cu12 and nvidia-cusolver-cu12 install beside the package (the
# extra dft of MDAPY), else where the system has them. Their headers come from
# the toolkit, and their sonames from the files of the toolkit, as the build
# finds them. The wheel holds none of them.
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
  # the soname of each library the module loads, from its file: the name up to
  # its major version, as libcublas.so.12 of libcublas.so.12.9.1.4
  set(MDAPY_DFT_SONAMES "")
  foreach(pair "CUBLAS;${CUDA_cublas_LIBRARY}" "CUBLASLT;${CUDA_cublasLt_LIBRARY}" "CUFFT;${CUDA_cufft_LIBRARY}"
               "CUSOLVER;${CUDA_cusolver_LIBRARY}" "CUSPARSE;${MDAPY_cusparse_LIBRARY}"
               "NVJITLINK;${MDAPY_nvJitLink_LIBRARY}")
    list(GET pair 0 name)
    list(GET pair 1 file)
    if(NOT EXISTS "${file}")
      message(FATAL_ERROR "the DFT needs the library ${name} of the CUDA toolkit, not found")
    endif()
    get_filename_component(real "${file}" REALPATH)
    get_filename_component(real "${real}" NAME)
    string(REGEX MATCH "^lib[^.]+\\.so\\.[0-9]+" soname "${real}")
    if(NOT soname)
      message(FATAL_ERROR "no soname in the file ${real} of the library ${name}")
    endif()
    list(APPEND MDAPY_DFT_SONAMES "MDAPY_SONAME_${name}=\"${soname}\"")
  endforeach()
  message(STATUS "The DFT loads ${MDAPY_DFT_SONAMES}")
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
  target_link_libraries(mdapy_dft PUBLIC CUDA::cudart_static ${CMAKE_DL_LIBS})
  # the headers of the libraries loaded at their first use, not the libraries
  target_include_directories(mdapy_dft SYSTEM PUBLIC $<TARGET_PROPERTY:CUDA::cublas,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:CUDA::cufft,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:CUDA::cusolver,INTERFACE_INCLUDE_DIRECTORIES>)
  target_compile_definitions(mdapy_dft PUBLIC ${MDAPY_DFT_SONAMES})
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
if(MDAPY_WERROR)
  target_compile_options(_dft PRIVATE -Werror)
  target_compile_options(mdapy_dft PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
install(TARGETS _dft LIBRARY DESTINATION ${MDAPY_DFT_PACKAGE})

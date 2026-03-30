cmake_minimum_required(VERSION 3.15...3.27)
project(mdapy LANGUAGES CXX C)   # 加 C：Tachyon 源码是纯 C

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if(APPLE)
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
  set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
endif()

find_package(OpenMP COMPONENTS CXX REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/voro++/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/NEPCPU)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/ptm)

file(GLOB PTM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/extern/ptm/ptm_*.cpp)
file(GLOB NEP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/extern/NEPCPU/*.cpp)

nanobind_add_module(_neighbor         STABLE_ABI src/neighbor.cpp)
nanobind_add_module(_structure_entropy STABLE_ABI src/structure_entropy.cpp)
nanobind_add_module(_voronoi          STABLE_ABI src/voronoi.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/extern/voro++/src/voro++.cc)
nanobind_add_module(_nepcal           STABLE_ABI src/neppy.cpp ${NEP_SOURCES})
nanobind_add_module(_repeat_cell      STABLE_ABI src/repeat_cell.cpp)
nanobind_add_module(_aabbtree         STABLE_ABI src/aabb_tree.cpp)
nanobind_add_module(_cluster          STABLE_ABI src/cluster.cpp)
nanobind_add_module(_spline           STABLE_ABI src/spline.cpp)
nanobind_add_module(_cna              STABLE_ABI src/cna.cpp)
nanobind_add_module(_polycrystal      STABLE_ABI src/polycrystal.cpp)
nanobind_add_module(_ptm              STABLE_ABI
    src/polyhedral_template_matching.cpp ${PTM_SOURCES})
nanobind_add_module(_eam              STABLE_ABI src/eam.cpp)
nanobind_add_module(_sbo              STABLE_ABI src/steinhardt_bond_orientation.cpp)
nanobind_add_module(_rdf              STABLE_ABI src/radial_distribution_function.cpp)
nanobind_add_module(_sfc              STABLE_ABI src/structure_factor.cpp)
nanobind_add_module(_csp              STABLE_ABI src/centro_symmetry_parameter.cpp)
nanobind_add_module(_wcp              STABLE_ABI src/warren_cowley_parameter.cpp)
nanobind_add_module(_strain           STABLE_ABI src/atomic_strain.cpp)
nanobind_add_module(_fccpft           STABLE_ABI src/identify_fcc_planar_faults.cpp)
nanobind_add_module(_aja              STABLE_ABI src/ackland_jones_analysis.cpp)
nanobind_add_module(_cnp              STABLE_ABI src/common_neighbor_parameter.cpp)
nanobind_add_module(_split            STABLE_ABI src/split_file.cpp)
nanobind_add_module(_atomtemp         STABLE_ABI src/atomic_temperature.cpp)
nanobind_add_module(_lindemann        STABLE_ABI src/lindemann.cpp)
nanobind_add_module(_bond_analysis    STABLE_ABI src/bond_analysis.cpp)

set(MODULES
    _neighbor _structure_entropy _voronoi _nepcal _repeat_cell
    _aabbtree _cluster _spline _cna _polycrystal _ptm _eam _sbo
    _rdf _sfc _csp _wcp _strain _fccpft _aja _cnp _split
    _atomtemp _lindemann _bond_analysis
)

foreach(mod IN LISTS MODULES)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${mod} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(MSVC)
    target_compile_options(${mod} PRIVATE /openmp:llvm)
    if(${mod} STREQUAL "_cna")
      target_compile_options(${mod} PRIVATE /DMSVC)
    endif()
    if(${mod} STREQUAL "_nepcal")
      target_compile_options(${mod} PRIVATE /D_OPENMP)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${mod} PRIVATE -O3 -fopenmp)
    if(${mod} STREQUAL "_nepcal")
      target_compile_options(${mod} PRIVATE -D_OPENMP)
    endif()
  else()
    target_compile_options(${mod} PRIVATE -O3)
    if(${mod} STREQUAL "_nepcal")
      target_compile_options(${mod} PRIVATE -D_OPENMP)
    endif()
  endif()
  install(TARGETS ${mod} LIBRARY DESTINATION mdapy)
endforeach()

###############################################################################
# Tachyon 静态库
#
# 目录约定（你已经按此放置）：
#   extern/tachyon/src/    ←  tachyon 所有 *.c / *.h
###############################################################################
set(TACHYON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/extern/tachyon/src")

set(TACHYON_SOURCES
    ${TACHYON_SRC}/api.c
    ${TACHYON_SRC}/apigeom.c
    ${TACHYON_SRC}/apitrigeom.c
    ${TACHYON_SRC}/box.c
    ${TACHYON_SRC}/camera.c
    # cone.c 在 0.99.5 中已移除
    ${TACHYON_SRC}/coordsys.c
    ${TACHYON_SRC}/cylinder.c
    ${TACHYON_SRC}/extvol.c
    ${TACHYON_SRC}/global.c
    ${TACHYON_SRC}/grid.c
    ${TACHYON_SRC}/hash.c
    ${TACHYON_SRC}/imageio.c
    ${TACHYON_SRC}/imap.c
    ${TACHYON_SRC}/intersect.c
    ${TACHYON_SRC}/jpeg.c
    ${TACHYON_SRC}/light.c
    ${TACHYON_SRC}/parallel.c
    # parvol.c 依赖 TACHYON_INTERNAL 内部结构体，无法在库外编译，已排除
    ${TACHYON_SRC}/plane.c
    ${TACHYON_SRC}/pngfile.c
    ${TACHYON_SRC}/ppm.c
    ${TACHYON_SRC}/psd.c
    ${TACHYON_SRC}/quadric.c
    ${TACHYON_SRC}/render.c
    ${TACHYON_SRC}/ring.c
    ${TACHYON_SRC}/sgirgb.c
    ${TACHYON_SRC}/shade.c
    ${TACHYON_SRC}/sphere.c
    ${TACHYON_SRC}/texture.c
    ${TACHYON_SRC}/tgafile.c
    ${TACHYON_SRC}/threads.c
    ${TACHYON_SRC}/trace.c
    ${TACHYON_SRC}/triangle.c
    ${TACHYON_SRC}/ui.c
    ${TACHYON_SRC}/util.c
    ${TACHYON_SRC}/vector.c
    ${TACHYON_SRC}/vol.c
    ${TACHYON_SRC}/winbmp.c
)
# 去重（ppm.c 上面写了两次，CMake 会自动去重，此处保持明确）
list(REMOVE_DUPLICATES TACHYON_SOURCES)

add_library(TachyonLib STATIC ${TACHYON_SOURCES})

target_compile_definitions(TachyonLib PRIVATE
    TACHYON_NO_DEPRECATED
    THR          # 开启 Tachyon 内置 POSIX 多线程
    _REENTRANT   # POSIX 线程安全
)

if(APPLE)
    target_compile_definitions(TachyonLib PRIVATE Bsd)
    target_link_libraries(TachyonLib PUBLIC pthread)
elseif(UNIX)
    target_compile_definitions(TachyonLib PRIVATE Linux)
    target_link_libraries(TachyonLib PUBLIC pthread)
elseif(WIN32)
    target_compile_definitions(TachyonLib PRIVATE WIN32)
    target_link_libraries(TachyonLib PUBLIC ws2_32)
endif()

# 暴露头文件目录（让 #include <tachyon.h> 生效）
target_include_directories(TachyonLib PUBLIC "${TACHYON_SRC}")
set_target_properties(TachyonLib PROPERTIES POSITION_INDEPENDENT_CODE ON)

###############################################################################
# _tachyon nanobind module  (CPU backend, always built)
###############################################################################
nanobind_add_module(_tachyon STABLE_ABI
    src/tachyon_render.cpp
)

target_include_directories(_tachyon PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"   # tachyon_render.h / tachyon_optix_render.h
    "${TACHYON_SRC}"                    # tachyon.h
)
target_link_libraries(_tachyon PRIVATE TachyonLib)

if(MSVC)
    target_compile_options(_tachyon PRIVATE /O2)
else()
    target_compile_options(_tachyon PRIVATE -O3)
endif()

###############################################################################
# Optional GPU backend: TachyonOptiX (NVIDIA OptiX 7 + CUDA)
#
# The CPU backend is ALWAYS compiled regardless of this block.
# The GPU backend is an opt-in addition that requires:
#   1. CUDA Toolkit on PATH (nvcc, cuda_runtime.h, libcudart)
#   2. OptiX 7 headers in extern/optix/include/
#      → git submodule add https://github.com/NVIDIA/optix-dev extern/optix
#
# OptiX 7+ is a header-only API; the implementation is in the NVIDIA display
# driver (libnvoptix.so / nvoptix.dll) which ships with any CUDA-capable GPU.
# No separate OptiX SDK download or NVIDIA account is needed.
#
# To enable GPU backend:
#   1. git submodule add https://github.com/NVIDIA/optix-dev extern/optix
#   2. pip install -e .          (CUDA toolkit must be on PATH)
#
# To disable GPU backend explicitly:
#   cmake -DMDAPY_OPTIX=OFF ..
###############################################################################
option(MDAPY_OPTIX "Build GPU backend using NVIDIA OptiX 7 (requires CUDA)" ON)

if(MDAPY_OPTIX)
    # --- Check 1: extern/optix/include/optix.h must exist ---
    # This is the ONLY accepted location. No fallback to system paths.
    # Run:  git submodule add https://github.com/NVIDIA/optix-dev extern/optix
    set(_optix_header "${CMAKE_CURRENT_SOURCE_DIR}/extern/optix/include/optix.h")

    if(NOT EXISTS "${_optix_header}")
        message(STATUS "[mdapy] extern/optix/include/optix.h not found — "
                       "GPU backend disabled. "
                       "To enable: git submodule add "
                       "https://github.com/NVIDIA/optix-dev extern/optix")
    else()
        # --- Check 2: CUDA Toolkit ---
        find_package(CUDAToolkit QUIET)

        if(NOT CUDAToolkit_FOUND)
            message(STATUS "[mdapy] CUDA toolkit not found — "
                           "GPU backend disabled.")
        else()
            # --- Check 3: OptiX version must be 7.x ---
            # TachyonOptiX.cu / TachyonOptiXShaders.cu were written for OptiX 7.
            # OptiX 8+ removed several exception codes used in the shaders.
            # Read OPTIX_VERSION from the header (format: major*10000 + minor*100 + patch).
            file(READ "${CMAKE_CURRENT_SOURCE_DIR}/extern/optix/include/optix.h"
                 _optix_h_content)
            string(REGEX MATCH "define OPTIX_VERSION[ 	]+([0-9]+)"
                   _optix_ver_match "${_optix_h_content}")
            set(_optix_version_num "${CMAKE_MATCH_1}")

            if(NOT _optix_version_num)
                # Fallback: try optix_types.h
                file(READ "${CMAKE_CURRENT_SOURCE_DIR}/extern/optix/include/optix_types.h"
                     _optix_t_content)
                string(REGEX MATCH "OPTIX_VERSION[ 	]+([0-9]+)"
                       _optix_ver_match "${_optix_t_content}")
                set(_optix_version_num "${CMAKE_MATCH_1}")
            endif()

            # OPTIX_VERSION 70600 = 7.6.0,  70700 = 7.7.0,  80000 = 8.0.0
            # Tachyon 0.99.5 requires OptiX 7.4-7.6 exactly:
            #   7.7 renamed optixModuleCreateFromPTX -> optixModuleCreate
            #   7.7 removed OptixPipelineLinkOptions::debugLevel
            #   8.0+ removed more exception codes
            if(_optix_version_num AND _optix_version_num GREATER_EQUAL 70700)
                math(EXPR _optix_major "${_optix_version_num} / 10000")
                math(EXPR _optix_minor "(${_optix_version_num} % 10000) / 100")
                message(FATAL_ERROR
                    "[mdapy] OptiX ${_optix_major}.${_optix_minor} detected in "
                    "extern/optix/include, but Tachyon 0.99.5 requires OptiX 7.x.\n"
                    "  Tachyon 0.99.5 requires OptiX 7.4, 7.5, or 7.6 exactly.\n"
                    "  Please switch to the OptiX 7.6 tag (last compatible version):\n"
                    "    cd extern/optix && git checkout tags/v7.6.0\n"
                    "  Or re-clone with the correct tag:\n"
                    "    rm -rf extern/optix\n"
                    "    git clone --branch v7.6.0 --depth 1 "
                    "https://github.com/NVIDIA/optix-dev extern/optix")
            elseif(_optix_version_num)
                math(EXPR _optix_major "${_optix_version_num} / 10000")
                math(EXPR _optix_minor "(${_optix_version_num} % 10000) / 100")
                if(_optix_version_num LESS 70400)
                    message(FATAL_ERROR
                        "[mdapy] OptiX ${_optix_major}.${_optix_minor} is too old. "
                        "Tachyon 0.99.5 requires OptiX 7.4–7.6. "
                        "Please use: git clone --branch v7.6.0 --depth 1 "
                        "https://github.com/NVIDIA/optix-dev extern/optix")
                else()
                    message(STATUS "[mdapy] OptiX ${_optix_major}.${_optix_minor} confirmed (7.4–7.6 required) ✓")
                endif()
            else()
                message(WARNING "[mdapy] Could not determine OptiX version — proceeding anyway.")
            endif()

            message(STATUS "[mdapy] CUDA ${CUDAToolkit_VERSION} found, "
                           "OptiX headers found — building GPU backend.")

            set(OPTIX_INCLUDE_DIR
                "${CMAKE_CURRENT_SOURCE_DIR}/extern/optix/include")

            enable_language(CUDA)

            # Compile TachyonOptiXShaders.cu → PTX (loaded at runtime by OptiX).
            set(OPTIX_SHADERS_CU "${TACHYON_SRC}/TachyonOptiXShaders.cu")
            set(OPTIX_SHADERS_PTX
                "${CMAKE_CURRENT_BINARY_DIR}/TachyonOptiXShaders.ptx")

            add_custom_command(
                OUTPUT  "${OPTIX_SHADERS_PTX}"
                COMMAND ${CMAKE_CUDA_COMPILER}
                        -ptx
                        -I "${TACHYON_SRC}"
                        -I "${OPTIX_INCLUDE_DIR}"
                        --generate-line-info
                        -arch sm_60   # minimum: Pascal (GTX 10xx and newer)
                        -o "${OPTIX_SHADERS_PTX}"
                        "${OPTIX_SHADERS_CU}"
                DEPENDS "${OPTIX_SHADERS_CU}"
                COMMENT "[mdapy] Compiling TachyonOptiX shaders to PTX"
            )
            add_custom_target(TachyonOptiXShadersPTX
                DEPENDS "${OPTIX_SHADERS_PTX}")

            # Add two CUDA source files to the _tachyon module:
            #   1. src/tachyon_optix_impl.cu  — our PIMPL implementation (mdapy code)
            #   2. extern/tachyon/src/TachyonOptiX.cu — Tachyon GPU engine
            #
            # CRITICAL: tachyon_render.cpp (compiled by g++) includes only the clean
            # tachyon_optix_render.h PIMPL interface and NEVER sees TachyonOptiX.h.
            # This avoids symbol clashes (rt_texture, RT_FOG_NONE, __saturatef, etc.)
            # between tachyon.h (CPU API) and TachyonOptiX.h (GPU internal API).
            target_sources(_tachyon PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/tachyon_optix_impl.cu"
                "${TACHYON_SRC}/TachyonOptiX.cu"
            )
            set_source_files_properties(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/tachyon_optix_impl.cu"
                "${TACHYON_SRC}/TachyonOptiX.cu"
                PROPERTIES LANGUAGE CUDA)

            target_include_directories(_tachyon PRIVATE "${OPTIX_INCLUDE_DIR}")
            target_link_libraries(_tachyon PRIVATE CUDA::cudart CUDA::cuda_driver dl)

            # These definitions are passed to ALL sources in _tachyon (both .cpp
            # and .cu).  The CUDA sources need TACHYONINTERNAL to avoid WKFUtils.h.
            target_compile_definitions(_tachyon PRIVATE
                MDAPY_OPTIX=1
                TACHYON_INTERNAL=1     # used by TachyonOptiX.cu itself
                TACHYONINTERNAL=1)     # used by TachyonOptiX.h → picks util.h
            # Explicitly undefine the denoiser feature: its OptiX 7.6 API changed
            # OptixDenoiserParams::denoiseAlpha from int to OptixDenoiserAlphaMode.
            target_compile_options(_tachyon PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-UTACHYON_OPTIXDENOISER>)

            set_target_properties(_tachyon PROPERTIES
                CUDA_SEPARABLE_COMPILATION OFF
                CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
                # 60=Pascal  70=Volta  75=Turing  80=Ampere  86=Ampere(RTX30)
                # 89=Ada(RTX40)  90=Hopper
            )

            add_dependencies(_tachyon TachyonOptiXShadersPTX)
            install(FILES "${OPTIX_SHADERS_PTX}" DESTINATION mdapy)

            message(STATUS "[mdapy] GPU backend ENABLED. "
                           "PTX → ${OPTIX_SHADERS_PTX}")

        endif()   # CUDAToolkit_FOUND
    endif()   # optix.h exists
else()
    message(STATUS "[mdapy] GPU backend disabled (MDAPY_OPTIX=OFF).")
endif()   # MDAPY_OPTIX

install(TARGETS _tachyon LIBRARY DESTINATION mdapy)
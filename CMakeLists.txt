cmake_minimum_required(VERSION 3.15...3.27)
project(mdapy LANGUAGES CXX C)   # 加 C：Tachyon 源码是纯 C

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# ---------------------------------------------------------------------------
# OpenMP discovery
#
# Strategy: when building inside a conda env ($CONDA_PREFIX set), link against
# whatever OpenMP runtime that env already provides. That way mdapy shares a
# single runtime with ovito / torch / sklearn installed in the same env —
# no duplicate libomp in the process, no "OMP Error #15", no segfaults.
#
# Outside a conda env (e.g. the cibuildwheel CI job that produces the PyPI
# wheels) we fall back to a platform default: Homebrew on macOS, system
# libgomp on Linux, MSVC's bundled libomp on Windows. Those wheels are
# self-contained (delocate / auditwheel / delvewheel bundle what's needed),
# so end users installing the wheel still get a working setup.
# ---------------------------------------------------------------------------
if(APPLE)
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/libomp.dylib")
    set(OpenMP_omp_LIBRARY "$ENV{CONDA_PREFIX}/lib/libomp.dylib")
    set(OpenMP_CXX_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
  else()
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    set(OpenMP_CXX_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
  endif()
elseif(UNIX)
  # Linux: GCC's -fopenmp auto-links libgomp.so.1 from the toolchain's
  # runtime. To make the installed .so files resolve libgomp at runtime to
  # the conda env's copy (matching whatever ovito / torch / sklearn loaded),
  # add $CONDA_PREFIX/lib to the RPATH. Conda-forge's llvm-openmp also
  # installs libomp.so here, so this covers both the GCC-libgomp and the
  # clang-libomp cases without branching on the compiler.
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
    list(APPEND CMAKE_BUILD_RPATH   "$ENV{CONDA_PREFIX}/lib")
    list(APPEND CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  endif()
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
nanobind_add_module(_fast_knn         STABLE_ABI src/fast_knn.cpp)
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
nanobind_add_module(_build_bond       STABLE_ABI src/build_bond.cpp)
nanobind_add_module(_sqs              STABLE_ABI src/sqs.cpp)

set(MODULES
    _neighbor _structure_entropy _voronoi _nepcal _repeat_cell
    _fast_knn _cluster _spline _cna _polycrystal _ptm _eam _sbo
    _rdf _sfc _csp _wcp _strain _fccpft _aja _cnp _split
    _atomtemp _lindemann _bond_analysis _build_bond _sqs
)

foreach(mod IN LISTS MODULES)
  if(MSVC)
    # On MSVC, use /openmp:llvm directly and skip OpenMP::OpenMP_CXX.
    # Linking OpenMP::OpenMP_CXX adds /openmp, then /openmp:llvm overrides it,
    # producing "cl: D9025: overriding /openmp with /openmp:llvm" for every module.
    # Using only /openmp:llvm avoids the warning entirely.
    target_compile_options(${mod} PRIVATE /openmp:llvm)
    # Suppress linker warnings about CRT symbol imports from mixed static libs.
    # LNK4217/LNK4286 are informational: a symbol defined in libucrt.lib is
    # imported by an object — harmless but very verbose.
    target_link_options(${mod} PRIVATE /ignore:4217 /ignore:4286 /ignore:4098)
    if(${mod} STREQUAL "_cna")
      target_compile_options(${mod} PRIVATE /DMSVC)
    endif()
    if(${mod} STREQUAL "_nepcal")
      target_compile_options(${mod} PRIVATE /D_OPENMP)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${mod} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(${mod} PRIVATE -O3 -fopenmp)
    if(${mod} STREQUAL "_nepcal")
      target_compile_options(${mod} PRIVATE -D_OPENMP)
    endif()
  else()
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${mod} PRIVATE OpenMP::OpenMP_CXX)
    endif()
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
    "${CMAKE_CURRENT_SOURCE_DIR}/src"          # tachyon_render.h / tachyon_optix_render.h
    "${TACHYON_SRC}"                           # tachyon.h
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/stb"   # stb_image.h / stb_image_write.h
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

            ###################################################################
            # IMPORTANT: Do NOT use enable_language(CUDA).
            #
            # On Windows, CMake's CUDA language requires the CUDA VS toolset
            # integration (CUDA xx.x.targets inside the VS installation).
            # When building via pip (isolated cmake without a full VS shell)
            # that toolset is never found and CMake aborts:
            #   "No CUDA toolset found."
            #
            # Solution: locate nvcc with find_program(), compile the .cu files
            # via add_custom_command(), and link the resulting object files.
            # This works on Linux, macOS, and Windows without any VS CUDA toolset.
            ###################################################################

            # find_program locates nvcc from the CUDAToolkit already found above.
            find_program(NVCC_EXECUTABLE nvcc
                HINTS "${CUDAToolkit_BIN_DIR}"
                REQUIRED)
            message(STATUS "[mdapy] nvcc: ${NVCC_EXECUTABLE}")

            # Include directories required by both .cu files.
            set(_CUDA_INC
                "${TACHYON_SRC}"
                "${OPTIX_INCLUDE_DIR}"
                "${CUDAToolkit_INCLUDE_DIRS}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src"
            )
            set(_NVCC_INCS "")
            foreach(_inc IN LISTS _CUDA_INC)
                list(APPEND _NVCC_INCS "-I${_inc}")
            endforeach()

            # GPU architecture targets.  Pascal/Volta (sm_60/70) were dropped
            # in CUDA 13; keep Turing (sm_75) and newer.
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                set(_GENCODE_FLAGS
                    -gencode arch=compute_75,code=sm_75
                    -gencode arch=compute_80,code=sm_80
                    -gencode arch=compute_86,code=sm_86
                    -gencode arch=compute_89,code=sm_89
                    -gencode arch=compute_90,code=sm_90
                )
                set(_PTX_ARCH "sm_75")
            else()
                set(_GENCODE_FLAGS
                    -gencode arch=compute_60,code=sm_60
                    -gencode arch=compute_70,code=sm_70
                    -gencode arch=compute_75,code=sm_75
                    -gencode arch=compute_80,code=sm_80
                    -gencode arch=compute_86,code=sm_86
                    -gencode arch=compute_89,code=sm_89
                    -gencode arch=compute_90,code=sm_90
                )
                set(_PTX_ARCH "sm_60")
            endif()

            # Common nvcc flags for compiling .cu -> relocatable object (-dc).
            # -dc emits device code that must be device-linked before host linking.
            if(WIN32)
                set(_NVCC_FLAGS
                    -dc --generate-line-info
                    ${_GENCODE_FLAGS}
                    -DMDAPY_OPTIX=1 -DTACHYON_INTERNAL=1 -DTACHYONINTERNAL=1
                    -UTACHYON_OPTIXDENOISER
                    # On Windows, `uint` is not a built-in type (comes from
                    # POSIX <sys/types.h> which MSVC does not include automatically).
                    # Map it via a macro. Also map strcasecmp -> _stricmp (POSIX vs MSVC).
                    "-Duint=unsigned int"
                    "-Dstrcasecmp=_stricmp"
                    # _USE_MATH_DEFINES exposes M_PI under MSVC; /wd4819 silences
                    # the "file contains a character that cannot be represented"
                    # code-page warning emitted by nvcc on Windows.
                    # /MD: use dynamic CRT (DLL), matching cl.exe default.
                    # Mixing /MT (nvcc default) with /MD (cl.exe) causes
                    # LNK2038 "RuntimeLibrary mismatch" at link time.
                    # /wd4819: suppress code-page warning from nvcc on Windows.
                    # /wd4244: suppress narrowing conversion warnings.
                    -Xcompiler "/MD /D_USE_MATH_DEFINES /wd4819 /wd4244"
                    # Suppress nvcc warning about old GPU arch (sm_60..74)
                    # being deprecated in future CUDA releases.
                    -Wno-deprecated-gpu-targets
                )
                set(_OBJ_EXT "obj")
            else()
                set(_NVCC_FLAGS
                    -dc --generate-line-info
                    # manylinux_2_28 ships GCC 13/14 while CUDA 12.x only officially
                    # supports up to GCC 12.  -allow-unsupported-compiler suppresses
                    # the version check.  -ccbin pins nvcc to the same host compiler
                    # that CMake uses (set via CC/CXX env vars in CI to gcc-12), so
                    # nvcc doesn't pick up gcc-14 from PATH whose STL headers
                    # (__bfloat16_t, _Float128, etc.) are incompatible with CUDA 12.3.
                    -allow-unsupported-compiler
                    -ccbin "${CMAKE_CXX_COMPILER}"
                    ${_GENCODE_FLAGS}
                    -DMDAPY_OPTIX=1 -DTACHYON_INTERNAL=1 -DTACHYONINTERNAL=1
                    -UTACHYON_OPTIXDENOISER
                    -Xcompiler -fPIC
                )
                set(_OBJ_EXT "o")
            endif()

            # ── PTX shader: TachyonOptiXShaders.cu → .ptx ─────────────────
            # (loaded at runtime by OptiX; compiled to PTX only, not an object)
            set(OPTIX_SHADERS_CU  "${TACHYON_SRC}/TachyonOptiXShaders.cu")
            set(OPTIX_SHADERS_PTX "${CMAKE_CURRENT_BINARY_DIR}/TachyonOptiXShaders.ptx")

            # Windows-only: `uint` is not built-in (no POSIX <sys/types.h>) and
            # strcasecmp is not available (use _stricmp). On Linux these macros
            # conflict with sys/types.h, so they must be omitted.
            if(WIN32)
                set(_PTX_PLATFORM_DEFS "-Duint=unsigned int" "-Dstrcasecmp=_stricmp")
            else()
                set(_PTX_PLATFORM_DEFS "")
            endif()

            add_custom_command(
                OUTPUT  "${OPTIX_SHADERS_PTX}"
                COMMAND "${NVCC_EXECUTABLE}"
                        -ptx --generate-line-info
                        -allow-unsupported-compiler
                        -ccbin "${CMAKE_CXX_COMPILER}"
                        -arch ${_PTX_ARCH}
                        ${_NVCC_INCS}
                        -DTACHYON_INTERNAL=1 -DTACHYONINTERNAL=1
                        -UTACHYON_OPTIXDENOISER
                        ${_PTX_PLATFORM_DEFS}
                        -Wno-deprecated-gpu-targets
                        -o "${OPTIX_SHADERS_PTX}"
                        "${OPTIX_SHADERS_CU}"
                DEPENDS "${OPTIX_SHADERS_CU}"
                COMMENT "[mdapy] nvcc: TachyonOptiXShaders.cu -> PTX"
            )
            add_custom_target(TachyonOptiXShadersPTX ALL
                DEPENDS "${OPTIX_SHADERS_PTX}")

            # ── TachyonOptiX.cu → relocatable object ──────────────────────
            set(_TACHYON_OPTIX_CU  "${TACHYON_SRC}/TachyonOptiX.cu")
            set(_TACHYON_OPTIX_OBJ "${CMAKE_CURRENT_BINARY_DIR}/TachyonOptiX.${_OBJ_EXT}")

            add_custom_command(
                OUTPUT  "${_TACHYON_OPTIX_OBJ}"
                COMMAND "${NVCC_EXECUTABLE}"
                        ${_NVCC_FLAGS} ${_NVCC_INCS}
                        -o "${_TACHYON_OPTIX_OBJ}"
                        "${_TACHYON_OPTIX_CU}"
                DEPENDS "${_TACHYON_OPTIX_CU}"
                        "${TACHYON_SRC}/TachyonOptiX.h"
                        "${TACHYON_SRC}/TachyonOptiXShaders.h"
                COMMENT "[mdapy] nvcc: TachyonOptiX.cu -> object"
            )

            # ── tachyon_optix_impl.cu → relocatable object ─────────────────
            set(_IMPL_CU  "${CMAKE_CURRENT_SOURCE_DIR}/src/tachyon_optix_impl.cu")
            set(_IMPL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/tachyon_optix_impl.${_OBJ_EXT}")

            add_custom_command(
                OUTPUT  "${_IMPL_OBJ}"
                COMMAND "${NVCC_EXECUTABLE}"
                        ${_NVCC_FLAGS} ${_NVCC_INCS}
                        -o "${_IMPL_OBJ}"
                        "${_IMPL_CU}"
                DEPENDS "${_IMPL_CU}"
                        "${TACHYON_SRC}/TachyonOptiX.h"
                        "${CMAKE_CURRENT_SOURCE_DIR}/src/tachyon_optix_render.h"
                COMMENT "[mdapy] nvcc: tachyon_optix_impl.cu -> object"
            )

            # ── Device-link the two relocatable objects ────────────────────
            # When -dc is used, a separate nvcc -dlink pass is required to merge
            # the device code before the host linker sees the final .obj files.
            set(_DLINK_OBJ "${CMAKE_CURRENT_BINARY_DIR}/tachyon_optix_dlink.${_OBJ_EXT}")
            if(WIN32)
                set(_DLINK_EXTRA "")
            else()
                set(_DLINK_EXTRA -Xcompiler -fPIC)
            endif()

            add_custom_command(
                OUTPUT  "${_DLINK_OBJ}"
                COMMAND "${NVCC_EXECUTABLE}"
                        -dlink
                        ${_GENCODE_FLAGS}
                        ${_DLINK_EXTRA}
                        -o "${_DLINK_OBJ}"
                        "${_TACHYON_OPTIX_OBJ}" "${_IMPL_OBJ}"
                DEPENDS "${_TACHYON_OPTIX_OBJ}" "${_IMPL_OBJ}"
                COMMENT "[mdapy] nvcc: device-link CUDA objects"
            )

            # Custom target ensures obj files are built before _tachyon links.
            add_custom_target(TachyonOptiXBuild ALL
                DEPENDS "${_TACHYON_OPTIX_OBJ}" "${_IMPL_OBJ}" "${_DLINK_OBJ}")

            # Tell tachyon_render.cpp (compiled by cl.exe/g++) that OptiX is
            # available, so it exposes TachyonOptiXRenderer and has_optix()=true.
            target_compile_definitions(_tachyon PRIVATE MDAPY_OPTIX=1)
            # tachyon_render.h uses M_PI. MSVC only exposes it with this define.
            if(MSVC)
                target_compile_definitions(_tachyon PRIVATE _USE_MATH_DEFINES)
            endif()

            target_include_directories(_tachyon PRIVATE "${OPTIX_INCLUDE_DIR}")

            # Inject the three .obj/.o files directly into the link command.
            #
            # target_link_libraries() with raw paths is unreliable on the MSVC
            # generator — CMake may silently discard them instead of passing them
            # to link.exe.  target_link_options() bypasses CMake's translation
            # layer and appends the paths verbatim to the linker command line:
            #   - MSVC link.exe accepts bare .obj paths on its command line.
            #   - GNU ld / lld accept bare .o paths equally.
            # This is the most reliable way to force-include pre-built objects.
            if(WIN32)
                target_link_options(_tachyon PRIVATE
                    # CUDA objects: TachyonOptiX engine + our PIMPL + device-link stub
                    "${_TACHYON_OPTIX_OBJ}"
                    "${_IMPL_OBJ}"
                    "${_DLINK_OBJ}"
                    # Suppress noisy MSVC linker informational warnings.
                    /ignore:4098   # defaultlib conflict (CRT mismatch notice)
                    /ignore:4217   # symbol imported by object (harmless)
                    /ignore:4286   # symbol imported by multiple objects (harmless)
                )
                # cudart_static: statically link CUDA Runtime so cudart64_1xx.dll
                # is not a runtime dependency of the wheel.  Only nvcuda.dll (the
                # NVIDIA display driver, always present on GPU machines) is needed.
                target_link_libraries(_tachyon PRIVATE
                    CUDA::cudart_static CUDA::cuda_driver)
            else()
                target_link_options(_tachyon PRIVATE
                    "${_TACHYON_OPTIX_OBJ}"
                    "${_IMPL_OBJ}"
                    "${_DLINK_OBJ}"
                )
                # cudart_static: statically link CUDA Runtime so libcudart.so is
                # not bundled in the manylinux wheel.  Only libcuda.so.1 (the NVIDIA
                # driver stub, always present on GPU machines) is needed at runtime.
                target_link_libraries(_tachyon PRIVATE
                    CUDA::cudart_static CUDA::cuda_driver dl)
            endif()

            add_dependencies(_tachyon TachyonOptiXBuild TachyonOptiXShadersPTX)
            install(FILES "${OPTIX_SHADERS_PTX}" DESTINATION mdapy)

            message(STATUS "[mdapy] GPU backend ENABLED (nvcc custom-command). "
                           "PTX: ${OPTIX_SHADERS_PTX}")

        endif()   # CUDAToolkit_FOUND
    endif()   # optix.h exists
else()
    message(STATUS "[mdapy] GPU backend disabled (MDAPY_OPTIX=OFF).")
endif()   # MDAPY_OPTIX

install(TARGETS _tachyon LIBRARY DESTINATION mdapy)

cmake_minimum_required(VERSION 3.15...3.27)
project(mdapy LANGUAGES CXX)

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
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/voro++/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/NEPCPU)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/NEPCPU/qnep)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/ptm)

file(GLOB PTM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/ptm/ptm_*.cpp
)
file(GLOB qNEP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/NEPCPU/qnep/*.cpp
)

nanobind_add_module(_neighbor STABLE_ABI src/neighbor.cpp)
nanobind_add_module(_structure_entropy STABLE_ABI src/structure_entropy.cpp)
nanobind_add_module(_voronoi STABLE_ABI src/voronoi.cpp ${CMAKE_CURRENT_SOURCE_DIR}/extern/voro++/src/voro++.cc)
nanobind_add_module(_nepcal STABLE_ABI src/neppy.cpp ${CMAKE_CURRENT_SOURCE_DIR}/extern/NEPCPU/nep.cpp)
nanobind_add_module(_qnepcal STABLE_ABI src/qneppy.cpp ${qNEP_SOURCES})
nanobind_add_module(_repeat_cell STABLE_ABI src/repeat_cell.cpp)
nanobind_add_module(_aabbtree STABLE_ABI src/aabb_tree.cpp)
nanobind_add_module(_cluster STABLE_ABI src/cluster.cpp)
nanobind_add_module(_spline STABLE_ABI src/spline.cpp)
nanobind_add_module(_cna STABLE_ABI src/cna.cpp)
nanobind_add_module(_polycrystal STABLE_ABI src/polycrystal.cpp)
nanobind_add_module(_ptm STABLE_ABI
    src/polyhedral_template_matching.cpp
    ${PTM_SOURCES}
)
nanobind_add_module(_eam STABLE_ABI src/eam.cpp)
nanobind_add_module(_sbo STABLE_ABI src/steinhardt_bond_orientation.cpp)
nanobind_add_module(_rdf STABLE_ABI src/radial_distribution_function.cpp)
nanobind_add_module(_sfc STABLE_ABI src/structure_factor.cpp)
nanobind_add_module(_csp STABLE_ABI src/centro_symmetry_parameter.cpp)
nanobind_add_module(_wcp STABLE_ABI src/warren_cowley_parameter.cpp)
nanobind_add_module(_strain STABLE_ABI src/atomic_strain.cpp)
nanobind_add_module(_fccpft STABLE_ABI src/identify_fcc_planar_faults.cpp)
nanobind_add_module(_aja STABLE_ABI src/ackland_jones_analysis.cpp)
nanobind_add_module(_cnp STABLE_ABI src/common_neighbor_parameter.cpp)
nanobind_add_module(_split STABLE_ABI src/split_file.cpp)
nanobind_add_module(_atomtemp STABLE_ABI src/atomic_temperature.cpp)
nanobind_add_module(_lindemann STABLE_ABI src/lindemann.cpp)
nanobind_add_module(_bond_analysis STABLE_ABI src/bond_analysis.cpp)
set(MODULES
    _neighbor
    _structure_entropy
    _voronoi
    _nepcal
    _qnepcal
    _repeat_cell
    _aabbtree
    _cluster
    _spline
    _cna
    _polycrystal
    _ptm
    _eam
    _sbo
    _rdf
    _sfc
    _csp
    _wcp
    _strain
    _fccpft
    _aja
    _cnp
    _split
    _atomtemp
    _lindemann
    _bond_analysis
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
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${mod} PRIVATE -O3 -fopenmp)
  else()
    target_compile_options(${mod} PRIVATE -O3)
  endif()

  install(TARGETS ${mod} LIBRARY DESTINATION mdapy)
endforeach()
#######################################################################################
#
#  Copyright 2025 OVITO GmbH, Germany
#
#  This file is part of OVITO (Open Visualization Tool).
#
#  OVITO is free software; you can redistribute it and/or modify it either under the
#  terms of the GNU General Public License version 3 as published by the Free Software
#  Foundation (the "GPL") or, at your option, under the terms of the MIT License.
#  If you do not alter this notice, a recipient may use your version of this
#  file under either the GPL or the MIT License.
#
#  You should have received a copy of the GPL along with this program in a
#  file LICENSE.GPL.txt.  You should have received a copy of the MIT License along
#  with this program in a file LICENSE.MIT.txt
#
#  This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND,
#  either express or implied. See the GPL or the MIT License for the specific language
#  governing rights and limitations.
#
#######################################################################################

IF(WIN32)
    # Always build this module as a static library under Windows, because PTM does not explicitly export any symbols.
    # Building as a DLL would require tagging all symbols to be exported with __declspec(dllexport) in the source  code.
    SET(BUILD_SHARED_LIBS OFF)
ENDIF()

# Build library.
ADD_LIBRARY(PolyhedralTemplateMatching
    ptm_alloy_types.cpp
    ptm_canonical_coloured.cpp
    ptm_convex_hull_incremental.cpp
    ptm_correspondences.cpp
    ptm_deformation_gradient.cpp
    ptm_graph_data.cpp
    ptm_graph_tools.cpp
    ptm_index.cpp
    ptm_initialize_data.cpp
    ptm_map_templates.cpp
    ptm_multishell.cpp
    ptm_neighbour_ordering.cpp
    ptm_normalize_vertices.cpp
    ptm_polar.cpp
    ptm_quat.cpp
    ptm_solid_angles.cpp
    ptm_structure_matcher.cpp
    ptm_voronoi_cell.cpp
)

# Simply export all symbols defined by the PTM library, because the code doesn't explicitly export certain API symbols.
SET_TARGET_PROPERTIES(PolyhedralTemplateMatching PROPERTIES CXX_VISIBILITY_PRESET "default")

# Give our library file a new name to not confuse it with any system versions of the library.
SET_TARGET_PROPERTIES(PolyhedralTemplateMatching PROPERTIES OUTPUT_NAME "ovito_ptm")

# PTM library does not support unity build. Need to turn it off.
SET_TARGET_PROPERTIES(PolyhedralTemplateMatching PROPERTIES UNITY_BUILD OFF)

# Link to Voro++ library.
TARGET_LINK_LIBRARIES(PolyhedralTemplateMatching PRIVATE VoroPlusPlus)

# Make header files of this library available to dependent targets.
TARGET_INCLUDE_DIRECTORIES(PolyhedralTemplateMatching INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/..")

IF(EMSCRIPTEN)
    # Adopt the Emscript compiler flags from Qt's platform target.
    # That's because all libraries linked into the static WASM module need be built with the same flags.
    TARGET_LINK_LIBRARIES(PolyhedralTemplateMatching PRIVATE Qt::Platform)
ENDIF()

# Add standard compile and link options to the CMake target.
OVITO_ADD_STANDARD_COMPILE_OPTIONS(PolyhedralTemplateMatching)

IF(NOT BUILD_SHARED_LIBS AND OVITO_BUILD_PYPI)
    # Since we will link this library into the dynamically loaded Python extension module, we need to use the fPIC flag.
    SET_PROPERTY(TARGET PolyhedralTemplateMatching PROPERTY POSITION_INDEPENDENT_CODE ON)
ENDIF()

# Export this target.
INSTALL(TARGETS PolyhedralTemplateMatching EXPORT OVITO
    RUNTIME DESTINATION "${OVITO_RELATIVE_LIBRARY_DIRECTORY}"
    LIBRARY DESTINATION "${OVITO_RELATIVE_LIBRARY_DIRECTORY}"
    ARCHIVE DESTINATION "${OVITO_RELATIVE_LIBRARY_DIRECTORY}" COMPONENT "development")

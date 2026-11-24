# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

# The sources of the GPU side, for CUDA (cuda.cmake) and for HIP (hip.cmake):
# the kernels, and the host code they share with mdapy._core.
file(GLOB MDAPY_GPU_KERNELS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
set(MDAPY_GPU_HOST_SOURCES
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

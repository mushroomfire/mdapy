cmake_minimum_required(VERSION 3.18...3.31)
project(mdapy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MDAPY_WERROR "Treat compiler warnings as errors" OFF)
set(MDAPY_SANITIZE "" CACHE STRING
  "Sanitizer build: 'address' (AddressSanitizer + UndefinedBehaviorSanitizer) or 'thread' (ThreadSanitizer)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# SKBUILD_SABI_COMPONENT is set by scikit-build-core when building an abi3 wheel.
find_package(Python 3.10 REQUIRED
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

# A single extension module, mdapy._core, holds all compiled code. Each
# functional area registers its own submodule from src/cpp/bindings/.
file(GLOB_RECURSE MDAPY_CPP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp")

# NB_SUPPRESS_WARNINGS marks nanobind's headers as system headers, so the strict
# warning flags below (and -Werror) apply to mdapy's own code only.
nanobind_add_module(_core STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS ${MDAPY_CPP_SOURCES})
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")
target_link_libraries(_core PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(_core PRIVATE /W4 /permissive- /EHsc)
  if(MDAPY_WERROR)
    target_compile_options(_core PRIVATE /WX)
  endif()
else()
  # No fused multiply-add contraction: results are the same bits with every
  # compiler and processor.
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off)
  if(MDAPY_WERROR)
    target_compile_options(_core PRIVATE -Werror)
  endif()
  if(MDAPY_SANITIZE STREQUAL "address")
    target_compile_options(_core PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
    target_link_options(_core PRIVATE -fsanitize=address,undefined)
  elseif(MDAPY_SANITIZE STREQUAL "thread")
    target_compile_options(_core PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(_core PRIVATE -fsanitize=thread)
  elseif(NOT MDAPY_SANITIZE STREQUAL "")
    message(FATAL_ERROR "MDAPY_SANITIZE must be empty, 'address' or 'thread'")
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION mdapy)

#==--- ripple/apps/CMakeLists.txt -------------------------------------------==#
#
#                      Copyright (c) 2019 Rob Clucas
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ripple_SOURCE_DIR}/cmake)
include(ripple_cmake)

ripple_include_directories(
  ${CUDA_INCLUDE_DIR}
  ${ripple_SOURCE_DIR}/include
)

#==--- [applications] -------------------------------------------------------==#

ripple_add_executable(
  numa_arch ${ripple_SOURCE_DIR}/apps/numa_arch.cpp
)

#==--------------------------------------------------------------------------==#

# This creates all the targets, and is required to be called to enable the
# functionality which allows both CPU and GPU code to be compiled with either
# clang or nvcc + gcc.
# This needs to be called from the same directory in which the targets are
# created.
ripple_create_all_targets()


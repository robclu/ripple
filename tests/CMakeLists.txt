#==--- tests/CMakeLists.txt -------------------------------------------------==#
#
#                      Copyright (c) 2019 Streamline.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${streamline_SOURCE_DIR}/cmake)
include(streamline_cmake)

if (EXISTS ${streamline_SOURCE_DIR}/googletest)
  set(GTEST_INCLUDE  ${streamline_SOURCE_DIR}/googletest/googletest/include)
  set(GTEST_LIB_DIRS ${streamline_SOURCE_DIR}/lib)
  if (CMAKE_BUILD_TYPE EQUAL "Debug")
    set(GTEST_BOTH_LIBRARIES -lgtestd -lgtest_maind)
  else()
    set(GTEST_BOTH_LIBRARIES -lgtest -lgtest_main)
  endif()
else()
  set(GTEST_INCLUDE )
endif()

# Required for the experimental filesystem library.
set(LIBS -lstdc++-fs)

# All required cuda libraries.
if (APPLE)
  set(CUDA_LIBRARIES -lcudart -ldl -lrt -lcudadevrt -pthread)
else()
  set(CUDA_LIBRARIES -lcudart_static -ldl -lrt -lcudadevrt -pthread)
endif()

#fluid_add_definitions(${CMAKE_CUDA_FLAGS})
streamline_include_directories(
  ${CUDA_INCLUDE_DIR}
  ${streamline_SOURCE_DIR}/include
  ${GTEST_INCLUDE}
)
streamline_library_directories(${CUDA_LIB_DIRS} ${GTEST_LIB_DIRS})

#==--- Tests ----------------------------------------------------------------==#

streamline_add_executable(
  algorithm_tests ${streamline_SOURCE_DIR}/tests/algorithm_tests.cpp
)
streamline_target_link_libraries(
  algorithm_tests ${GTEST_BOTH_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

streamline_add_executable(
  array_tests ${streamline_SOURCE_DIR}/tests/array_tests.cpp
)
streamline_target_link_libraries(
  array_tests ${GTEST_BOTH_LIBRARIES}
)

#------------------------------------------------------------------------------#

streamline_add_executable(
  utility_tests ${streamline_SOURCE_DIR}/tests/utility_tests.cpp
)
streamline_target_link_libraries(
  utility_tests ${GTEST_BOTH_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

# This creates all the targets, and is required to be called to enable the
# functionality which allows both CPU and GPU code to be compiled with either
# clang or nvcc + gcc.
# This needs to be called from the same directory in which the targets are
# created.
streamline_create_all_targets()

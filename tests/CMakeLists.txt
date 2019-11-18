#==--- ripple/tests/CMakeLists.txt ------------------------------------------==#
#
#                      Copyright (c) 2019 Rob Clucas
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ripple_SOURCE_DIR}/cmake)
include(ripple_cmake)

if (EXISTS ${ripple_SOURCE_DIR}/googletest)
  set(GTEST_INCLUDE  ${ripple_SOURCE_DIR}/googletest/googletest/include)
  set(GTEST_LIB_DIRS ${ripple_SOURCE_DIR}/lib)
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

ripple_include_directories(
  ${CUDA_INCLUDE_DIR}
  ${ripple_SOURCE_DIR}/include
  ${GTEST_INCLUDE}
)
ripple_library_directories(${CUDA_LIB_DIRS} ${GTEST_LIB_DIRS})

#==--- [tests] --------------------------------------------------------------==#

ripple_add_executable(
  all_tests ${ripple_SOURCE_DIR}/tests/all.cpp
)
ripple_target_link_libraries(
  all_tests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

ripple_add_executable(
  algorithm_tests ${ripple_SOURCE_DIR}/tests/algorithm.cpp
)
ripple_target_link_libraries(
  algorithm_tests ${GTEST_BOTH_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

ripple_add_executable(
  container_tests ${ripple_SOURCE_DIR}/tests/container.cpp
)
ripple_target_link_libraries(
  container_tests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

ripple_add_executable(
  container_device_tests ${ripple_SOURCE_DIR}/tests/container_device.cu
)
ripple_target_link_libraries(
  container_device_tests ${GTEST_BOTH_LIBRARIES} ${CUDA_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

ripple_add_executable(
  execution_tests ${ripple_SOURCE_DIR}/tests/execution.cpp
)
ripple_target_link_libraries(
  execution_tests ${GTEST_BOTH_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

ripple_add_executable(
  iterator_tests ${ripple_SOURCE_DIR}/tests/iterator.cpp
)
ripple_target_link_libraries(
  iterator_tests ${GTEST_BOTH_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

ripple_add_executable(
  multidim_tests ${ripple_SOURCE_DIR}/tests/multidim.cpp
)
ripple_target_link_libraries(
  multidim_tests ${GTEST_BOTH_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

ripple_add_executable(
  storage_tests ${ripple_SOURCE_DIR}/tests/storage.cpp
)
ripple_target_link_libraries(
  storage_tests ${GTEST_BOTH_LIBRARIES}
)

#------------------------------------------------------------------------------#

ripple_add_executable(
  utility_tests ${ripple_SOURCE_DIR}/tests/utility.cpp
)
ripple_target_link_libraries(
  utility_tests ${GTEST_BOTH_LIBRARIES}
)

#==--------------------------------------------------------------------------==#

# This creates all the targets, and is required to be called to enable the
# functionality which allows both CPU and GPU code to be compiled with either
# clang or nvcc + gcc.
# This needs to be called from the same directory in which the targets are
# created.
ripple_create_all_targets()

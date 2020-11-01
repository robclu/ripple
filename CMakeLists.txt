#==--- ripple/CMakeLists.txt ------------------------------------------------==#
#
#                      Copyright (c) 2019 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ripple VERSION 0.1.0 LANGUAGES CXX)

#==--- [Build Type] ---------------------------------------------------------==#

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified!")
endif(NOT CMAKE_BUILD_TYPE)

#==--- [Compiler paths] -----------------------------------------------------==#

# 
if (APPLE)
  # Clang is the default for apple ...
else()
  set(CUDA_COMPILER_PATH  )
  set(GCC_COMPILER_PATH   )
  set(CLANG_COMPILER_PATH)
endif()

# Set clang usage variable for cuda:
if (NOT ${CUDA_CLANG})
  set(CUDA_CLANG FALSE)
  message(FATAL_ERROR "Only Clang is supported until nvcc adds c++17 support!")
else()
  set(CUDA_CLANG TRUE)
endif()

# Set clang usage variable for cxx:
if (NOT ${CXX_CLANG})
  set(CXX_CLANG FALSE)
  message(FATAL_ERROR "Only Clang is supported until nvcc adds c++17 support!")
else()
  set(CXX_CLANG TRUE)
endif()

message("-- Clang cuda compiler option : ${CUDA_CLANG}")
message("-- Clang cxx  compiler option : ${CXX_CLANG}")

if (EXISTS ${CLANG_PATH})
  set(CLANG_COMPILER_PATH "${CLANG_PATH}/bin/clang++")
endif()

# If clang for CXX then set the CXX compiler:
if (${CXX_CLANG})
  if (NOT EXISTS ${CLANG_PATH})
    message("-- CXX : Using system clang++ for cxx since CXX_CLANG=ON"     )
    message("         but CLANG_PATH is empty. Specify -DCLANG_PATH=<path>")
    message("         if a different clang is required."                   )
    set(RIPPLE_CXX_COMPILER clang++)
  else() 
    set(RIPPLE_CXX_COMPILER ${CLANG_COMPILER_PATH})
  endif()
endif()

# If clang for cuda then set the cuda compiler. When clang is used for cuda,
# the CXX compiler is automatically set to clang as well.
if (${CUDA_CLANG})
  if (NOT EXISTS ${CLANG_COMPILER_PATH})
    message("-- CUDA : Using system clang++ for cuda since -DCUDA_CLANG=ON"   )
    message("          but -DCLANG_PATH is empty. Specify -DCLANG_PATH=<path>")
    message("          if a different clang is required."                     )
    set(CLANG_COMPILER_PATH clang++)
  endif() 
  set(RIPPLE_CUDA_COMPILER ${CLANG_COMPILER_PATH})

  # If we use clang for cuda, we use clang for c++ as well
  set(RIPPLE_CXX_COMPILER ${CLANG_COMPILER_PATH})

  # We need to have a cuda path, if we don't we can't continue.
  if (NOT EXISTS ${CUDA_PATH})
    message("-- CUDA : CUDA_PATH is not spefified. Please specify cuda path")
    message("          with -DCUDA_PATH."                                   )
    message(FATAL_ERROR "-- CUDA : Can't continue without CUDA_PATH!"       )
  endif()

  set(USING_CLANG TRUE)
else()
  set(USING_CLANG FALSE)
endif()

# We definitely have a cuda path if we are here,
# so we can set the toolkit directory for cuda.
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})

# If not using clang, use NVCC.
if (NOT ${USING_CLANG})
  set(RIPPLE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  set(RIPPLE_CXX_COMPILER   ${STEAMLINE_CUDA_COMPILER})
endif()


# Test that the version of the cuda compiler is sufficient:
message("-- Using CUDA compiler : ${RIPPLE_CUDA_COMPILER}")
message("-- Using CXX  compiler : ${RIPPLE_CXX_COMPILER}" )
execute_process(
  COMMAND ${RIPPLE_CUDA_COMPILER} --version OUTPUT_VARIABLE NVCC_VERSION
)

if (NVCC_VERSION MATCHES "V9" OR NVCC_VERSION MATCHES "V10")
  # Nothing to do here ...
else()
  if (NOT ${USING_CLANG})
    message("-- ERROR: NVCC Version > 9 is required.")
    message(
      FATAL_ERROR "     Please set -DCUDA_PATH=<path/to/cuda/root> >= V9"
    )
  endif()
endif()

#==--- [Googletest] ---------------------------------------------------------==#

if (EXISTS ${ripple_SOURCE_DIR}/googletest)
  add_subdirectory(googletest)
else()
  message("-- NOTE: Googletest not found, fetching ...")
  execute_process(
    COMMAND git clone https://github.com/google/googletest.git 
      ${fluidity_SOURCE_DIR}/googletest
  )
  message("-- DONE: Fetched googletest.")
  add_subdirectory(googletest)
endif()

#==--- [Compiler Flags] -----------------------------------------------------==#

# Gtest does not currently build without this flag!
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CUDA_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/include)

if (APPLE)
  set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib)
else()
  set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

set(CUDA_DEFS "-DRIPPLE_CUDA_AVAILABLE=1")

list(LENGTH CUDA_ARCHS ARCH_COUNT)
if (${ARCH_COUNT} EQUAL "0")
  message("-- CUDA : No architecture version is set for cuda, using 3.5.")
  message("          Set using -DCUDA_ARCH_VER=<version> when building." )
  set(CUDA_ARCHS "35;70")
endif()

set(ARCH_LIST ${CUDA_ARCHS})
separate_arguments(ARCH_LIST)

foreach(ARCH IN LISTS ARCH_LIST)
  list(APPEND CUDA_ARCH_FLAGS)
  message("-- CUDA: Architecture added : ${ARCH}")
  if (NOT ${USING_CLANG})
    set(NEXT_ARCH "-gencode=arch=compute_${ARCH},code=sm_${ARCH}")
  else()
    set(NEXT_ARCH "--cuda-gpu-arch=sm_${ARCH}")
  endif()
  list(APPEND CUDA_ARCH_FLAGS ${NEXT_ARCH})

endforeach()

# Set max registers for kernels.
if (NOT MAX_REG_COUNT)
  set(MAX_REG_COUNT 64)
endif()

if (WIN32)

else(APPLE)
  set(FLAGS_ERRORS  "-Wpedantic -Wextra -Wall")
  set(FLAGS_PERF    "-O3 -std=c++17 -pthread -Wno-c++1z-extensions -fopenmp=libomp")
  set(FLAGS_DEBUG   "-g -G")
  set(DEBUG_FLAGS   "${FLAGS_PERF} ${FLAGS_ERRORS} ${FLAGS_DEBUG}")
  set(WARN_FLAGS    "-Wno-gnu-string-literal-operator-template")
  set(WARN_FLAGS    "${WARN_FLAGS} -Wno-unused-parameter")
  set(RELEASE_FLAGS "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WARN_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${WARN_FLAGS}")
  if (NOT ${USING_CLANG})
    set(CUDA_EXP_FLAGS "-expt-relaxed-constexpr -expt-extended-lambda")
    set(CUDA_FLAGS     "-use_fast_math -O3 -std=c++14 ${CUDA_EXP_FLAGS}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} -maxrregcount=${MAX_REG_COUNT}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} ${CUDA_ARCH_FLAG}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} -lineinfo -Xptxas=-v")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
    set(CUDA_FLAGS "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}") 
    set(CUDA_FLAGS "${CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
    set(CUDA_FLAGS "${CUDA_FLAGS} -O3 -std=c++17 -ffast-math -ffp-contract=fast")
    if (CMAKE_BUILD_TYPE MATCHES Release)
      set(CUDA_FLAGS "${CUDA_FLAGS}")
    else()
      set(CUDA_FLAGS "${CUDA_FLAGS} -g -Xcuda-ptxas -lineinfo -Xcuda-ptxas -v")
    endif()
  endif()
endif()
set(CUDA_FLAGS "${CUDA_FLAGS} ${CUDA_DEFS}")

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}" 
    CACHE STRING "CXX FLAGS" FORCE
  )
else()
  set(
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
    CACHE STRING "CXX FLAGS" FORCE
  )
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}")


#==--- [install] ------------------------------------------------------------==#

set(headers "")
file(
  GLOB_RECURSE 
  headers
  RELATIVE ${PROJECT_SOURCE_DIR} 
  "include/ripple/core/*.hpp" "include/ripple/core/*.cuh")

set(sources "")  
file(
  GLOB_RECURSE 
  sources 
  RELATIVE ${PROJECT_SOURCE_DIR} 
  "src/*.cpp" "src/*.cu")

# Currently this doesn't work because of the custom cuda build configuration.
# TODO: Change this to use built-in cmake cuda support now that it has been
#       updated.
add_library(ripple STATIC ${sources})
add_library(ripple::ripple ALIAS ripple)
target_include_directories(ripple PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(ripple SYSTEM PUBLIC
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)

# Setup package config
include(CMakePackageConfigHelpers)
set(RIPPLE_CMAKE_DIR "lib/cmake/ripple")
set(version_config "${CMAKE_CURRENT_BINARY_DIR}/ripple-config-version.cmake")
set(project_config "${CMAKE_CURRENT_BINARY_DIR}/ripple-config.cmake")
set(targets_export_name ripple-targets)

write_basic_package_version_file(
  ${version_config}
  VERSION       0.0.1
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/ripple-config.cmake.in"
  ${project_config}
  INSTALL_DESTINATION ${RIPPLE_CMAKE_DIR}
)

# Export targets for non-install use:
export(
  TARGETS   ripple 
  NAMESPACE ripple:: 
  FILE      ${CMAKE_CURRENT_BINARY_DIR}/${targets_export_name}.cmake
)

# Install version:
install(
  FILES ${project_config} ${version_config} DESTINATION ${RIPPLE_CMAKE_DIR}
)

# Install target:
install(TARGETS ripple DESTINATION lib EXPORT ${targets_export_name})
install(
  EXPORT      ${targets_export_name}
  DESTINATION lib
  NAMESPACE   ripple::
)
install(DIRECTORY include/ripple DESTINATION include)

#==--- [Module path] --------------------------------------------------------==#

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#==--- [Subdirectories] -----------------------------------------------------==#
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RIPPLE_BUILD_BENCHMARKS "build benchmark" OFF)
if (${RIPPLE_BUILD_BENCHMARKS} OR 
  (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    add_subdirectory(benchmarks)
endif()

option(RIPPLE_BUILD_DOCS "build docs" OFF)
if(${RIPPLE_BUILD_DOCS})
  add_subdirectory(docs)
endif()

option(RIPPLE_BUILD_EXAMPLES "build examples" ON)
if (${RIPPLE_BUILD_EXAMPLES} OR 
  (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR))
    add_subdirectory(examples)
endif()

string(
  REPLACE " " "\n+|                    " CXX_FLAG_STRING ${CMAKE_CXX_FLAGS})
string(
  REPLACE " " "\n+|                    " CUDA_FLAG_STRING ${CMAKE_CUDA_FLAGS})

message("
+|===== [build options] =======================================================|
+|
+| BUILD TYPE       : ${CMAKE_BUILD_TYPE}
+| CUDA_CLANG       : ${CUDA_CLANG}
+| CXX COMPILER     : ${RIPPLE_CXX_COMPILER}
+| CUDA COMPILER    : ${RIPPLE_CUDA_COMPILER}
+| CUDA PATH        : ${CUDA_TOOLKIT_ROOT_DIR}
+| CUDA INCLUDE     : ${CUDA_INCLUDE_DIR}
+| CUDA LIB         : ${CUDA_LIB_DIRS}
+| CUDA ARCHS       : ${ARCH_LIST}
+| BUILD_BENCHMARKS : ${RIPPLE_BUILD_BENCHMARKS}
+| BUILD_DOCS       : ${RIPPLE_BUILD_DOCS}
+| BUILD_EXAMPLES   : ${RIPPLE_BUILD_EXAMPLES}
+| CXX_FLAGS        : ${CXX_FLAG_STRING}
+| CUDA_FLAGS       : ${CUDA_FLAG_STRING}
+|
+|=============================================================================|
")
#==--- ripple/CMakeLists.txt ------------------------------------------------==#
#
#                      Copyright (c) 2019 Rob Clucas.
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ripple VERSION 0.1.0 LANGUAGES NONE)

#==--- [print options] ------------------------------------------------------==#

if (${PRINT_OPTIONS})
message("
+|===== [build options] =======================================================|
+| [required]
+| -DCMAKE_CUDA_COMPILER=  : Path to cuda compiler
+| -DCUDA_PATH=            : Path to CUDA toolkit root
+|
+| [optional]
+| -DPRINT_OPTIONS=           : {ON, OFF}
+| -DCMAKE_BUILD TYPE=        : {Debug, Release} 
+| -DCUDA_ARCHS=              : <arch>[;<arch>[;<arch>]]
+| -DMAXX_REG_COUNT=          : Max registers per kernel
+| -DCMAKE_CXX_COMPILER       : Path to host compiler
+| -DCMAKE_CUDA_HOST_COMPILER : Path to cuda host compiler
+| -DCMAKE_INSTALL_PREFIX=    : Path to install to
+| -DBUILD_APPS=              : {ON, OFF}
+| -DBUILD_TESTS=             : {ON, OFF}
+| -DBUILD_BENCHMARKS=        : {ON, OFF}
+| -DBUILD_DOCS=              : {ON, OFF}
+|
+|=============================================================================|
")
message(FATAL_ERROR "")
endif()

#==--- [Build Type] ---------------------------------------------------------==#

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified!")
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_STANDARD 17)

#==--- [Compiler setup] -----------------------------------------------------==#

if (NOT EXISTS ${CUDA_PATH})
  message("-- CUDA : CUDA_PATH is not spefified. Please specify cuda path")
  message("          with -DCUDA_PATH."                                   )
  message(FATAL_ERROR "-- CUDA : Can't continue without CUDA_PATH!"       )
endif()

set(CUDA_TOOLKIT_ROOT_DIR                  ${CUDA_PATH})
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDA_TOOLKIT_ROOT_DIR}/include)

# For some reason, cmake uses *this* variable to determine if the cuda compiler
# is valid, rather than the the toolkit root directory, so this hack fixes that.
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT ${CUDA_PATH})

# Execute cuda compiler to check what we are using:
execute_process(
  COMMAND ${CMAKE_CUDA_COMPILER} --version OUTPUT_VARIABLE VERSION)

if (VERSION MATCHES "clang")
  message("-- CUDA : Clang found as cuda compiler.")
  message("-- CUDA : Clang set as host compiler.")
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})
  set(CMAKE_CUDA_FLAGS "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")

  set(USING_CLANG TRUE)
  set(CXX_CLANG   TRUE)
  set(CUDA_CLANG  TRUE)
else()
  if (VERSION MATCHES "V11")
    message("-- CUDA : Nvcc set as cuda compiler.")
    message("-- CUDA : Using default host compiler.")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

    set(USING_CLANG 0)
    set(CXX_CLANG   )
    set(CUDA_CLANG  )
  else()
    message("-- CUDA : Only cuda version >= 11.0 if clang is not used as")
    message("          host and device compiler!")
    message(FATAL_ERROR)
 endif()
endif()

#set(CMAKE_CUDA_STANDARD          17)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

#==--- [Compiler Flags] -----------------------------------------------------==#

# Gtest does not currently build without this flag!
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CUDA_INCLUDE_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

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
  set(CUDA_ARCHS "80")
endif()

set(ARCH_LIST ${CUDA_ARCHS})
separate_arguments(ARCH_LIST)

foreach(ARCH IN LISTS ARCH_LIST)
  if (ARCH MATCHES "80")
    add_compile_definitions(RIPPLE_SM_80)
  endif()

  list(APPEND CUDA_ARCH_FLAGS)
  message("-- CUDA: Architecture added : ${ARCH}")
  if (NOT ${USING_CLANG})
    set(NEXT_ARCH "-gencode=arch=compute_${ARCH},code=sm_${ARCH}")
  else()
    set(NEXT_ARCH "--cuda-gpu-arch=sm_${ARCH} ")
  endif()
  list(APPEND CUDA_ARCH_FLAGS ${NEXT_ARCH})

endforeach()
set(CMAKE_CUDA_ARCHITECTURES ${ARCH_LIST})

add_compile_definitions(__STRICT_ANSI__=)

if (WIN32)

else(APPLE)
  set(FLAGS_ERRORS  "-Wpedantic -Wextra -Wall -I${ripple_SOURCE_DIR}/include")
  set(FLAGS_PERF    "-O3 -std=c++17 -pthread -Wno-c++1z-extensions")
  set(FLAGS_DEBUG   "-g")
  set(DEBUG_FLAGS   "${FLAGS_PERF} ${FLAGS_ERRORS} ${FLAGS_DEBUG}")
  set(WARN_FLAGS    "-Wno-gnu-string-literal-operator-template")
  set(WARN_FLAGS    "${WARN_FLAGS} -Wno-unused-parameter")
  set(RELEASE_FLAGS "${FLAGS_PERF} ${FLAGS_ERRORS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${WARN_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${WARN_FLAGS}")
  if (NOT ${USING_CLANG})
    set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -fdiagnostic-color=always")
    set(CUDA_EXP_FLAGS "-expt-relaxed-constexpr --extended-lambda")
    set(CUDA_FLAGS     "-use_fast_math -O3 --std c++17 ${CUDA_EXP_FLAGS}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
    set(CUDA_FLAGS     "${CUDA_FLAGS} -lineinfo -Xptxas=-v")
    set(CUDA_FLAGS     "${CUDA_FLAGS} -fdiagnostic-color=always")
    if (${MAX_REG_COUNT})
      set(CUDA_FLAGS "${CUDA_FLAGS} -maxrregcount=${MAX_REG_COUNT}")
    endif()
  else()
    set(CMAKE_CXX_FLAGS_RELEASE 
      "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -fcolor-diagnostics")
    set(CUDA_FLAGS "${CUDA_FLAGS} --no-cuda-version-check -fcolor-diagnostics")
    set(CUDA_FLAGS 
      "${CUDA_FLAGS} -O3 -std=c++17 -ffast-math -ffp-contract=fast")
    if (CMAKE_BUILD_TYPE MATCHES Release)
      set(CUDA_FLAGS "${CUDA_FLAGS} -Xcuda-ptxas -v")
    else()
      set(CUDA_FLAGS "${CUDA_FLAGS} -Xcuda-ptxas -lineinfo -v")
    endif()
  endif()
endif()

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
  set(CUDA_FLAGS "${CUDA_FLAGS} -G")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS}")

include_directories(${ripple_SOURCE_DIR}/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT ${CUDA_TOOLKIT_ROOT_DIR})

enable_language(CUDA)
enable_language(CXX)

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
add_library(ripple SHARED ${sources})
add_library(ripple::ripple ALIAS ripple)
target_include_directories(ripple PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
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

option(RIPPLE_BUILD_APPS "build apps" OFF)
if (${RIPPLE_BUILD_APPS})
  add_subdirectory(apps)
endif()

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

option(RIPPLE_BUILD_TESTS "build tests" OFF)
if(${RIPPLE_BUILD_TESTS})
  configure_file(
    CMakeLists-googletest.txt.in googletest-download/CMakeLists.txt
  )
  execute_process(
    COMMAND           ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE   result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(
    COMMAND           ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE   result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()
  
  # Add googletest directly to our build. This defines
  # the gtest and gtest_main targets.
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                   ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                   EXCLUDE_FROM_ALL)

  include_directories("${gtest_SOURCE_DIR}/include")

  add_subdirectory(tests)
endif()

string(
  REPLACE " " "\n+|                      " CXX_FLAG_STRING ${CMAKE_CXX_FLAGS})
string(
  REPLACE " " "\n+|                      " CUDA_FLAG_STRING ${CMAKE_CUDA_FLAGS})


message("
+|===== [build options] =======================================================|
+|
+| BUILD TYPE         : ${CMAKE_BUILD_TYPE}
+| CXX COMPILER       : ${CMAKE_CXX_COMPILER}
+| CUDA COMPILER      : ${CMAKE_CUDA_COMPILER}
+| CUDA HOST COMPILER : ${CMAKE_CUDA_HOST_COMPILER}
+| CUDA PATH          : ${CUDA_TOOLKIT_ROOT_DIR}
+| CUDA INCLUDE       : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
+| CUDA LIB           : ${CUDA_LIB_DIRS}
+| CUDA ARCHS         : ${ARCH_LIST}
+| BUILD_APPS         : ${RIPPLE_BUILD_APPS}
+| BUILD_BENCHMARKS   : ${RIPPLE_BUILD_BENCHMARKS}
+| BUILD_DOCS         : ${RIPPLE_BUILD_DOCS}
+| BUILD_EXAMPLES     : ${RIPPLE_BUILD_EXAMPLES}
+| BUILD_TESTS        : ${RIPPLE_BUILD_TESTS}
+| CXX_FLAGS          : ${CXX_FLAG_STRING}
+| CUDA_FLAGS         : ${CUDA_FLAG_STRING}
+|
+|=============================================================================|
")
#==--- ripple/benchmarks/CMakeLists.txt -------------------------------------==#
#
#                      Copyright (c) 2019 Rob Clucas
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

if (RIPPLE_BUILD_WITH_KOKKOS)
  add_subdirectory(kokkos)
  find_package(MPI REQUIRED)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

#==--- [force] --------------------------------------------------------------==#

add_executable(
  force ${ripple_SOURCE_DIR}/benchmarks/force/force.cu)
target_link_libraries(force ripple::ripple)

if (RIPPLE_BUILD_WITH_KOKKOS)
  add_executable(
    force_kokkos
    ${ripple_SOURCE_DIR}/benchmarks/force/force_kokkos.cu)
  target_link_libraries(
    force_kokkos
    Kokkos::kokkos 
    -lcudart
    ${MPI_CXX_LIBRARIES})
endif()

#=--- [particle update] -----------------------------------------------------==#

add_executable(
  particle_update ${ripple_SOURCE_DIR}/benchmarks/particle/particle_update.cu)
target_link_libraries(particle_update ripple::ripple -latomic)

if (RIPPLE_BUILD_WITH_KOKKOS)
  add_executable(
    particle_update_kokkos
    ${ripple_SOURCE_DIR}/benchmarks/particle/particle_update_kokkos.cpp)
  target_link_libraries(
    particle_update_kokkos
    Kokkos::kokkos 
    -lcudart
    ${MPI_CXX_LIBRARIES})
endif()

#=--- [reinitialization] ----------------------------------------------------==#

add_executable(
  reinitialization 
  ${ripple_SOURCE_DIR}/benchmarks/reinitialization/reinitialization.cu)
target_link_libraries(reinitialization ripple::ripple -latomic)

add_executable(
  reinitialization_original_fim
  ${ripple_SOURCE_DIR}/benchmarks/reinitialization/reinit_original_fim.cu)
target_link_libraries(reinitialization ripple::ripple -latomic)

if (RIPPLE_BUILD_WITH_KOKKOS)
  add_executable(
    reinit_kokkos
    ${ripple_SOURCE_DIR}/benchmarks/reinitialization/reinit_kokkos.cu)
  target_link_libraries(
    reinit_kokkos
    Kokkos::kokkos 
    -lcudart
    ${MPI_CXX_LIBRARIES})
endif()

#==--- [saxpy ---------------------------------------------------------------==#

add_executable(
  saxpy_cublas ${ripple_SOURCE_DIR}/benchmarks/saxpy/saxpy_cublas.cu)
target_link_libraries(saxpy_cublas ripple::ripple -lcublas)

add_executable(
  saxpy_ripple ${ripple_SOURCE_DIR}/benchmarks/saxpy/saxpy_ripple.cu)
target_link_libraries(saxpy_ripple ripple::ripple -latomic)

if (RIPPLE_BUILD_WITH_KOKKOS)
  add_executable(
    saxpy_kokkos
    ${ripple_SOURCE_DIR}/benchmarks/saxpy/saxpy_kokkos.cpp)
  target_link_libraries(
    saxpy_kokkos
    Kokkos::kokkos 
    -lcudart
    ${MPI_CXX_LIBRARIES})
endif()
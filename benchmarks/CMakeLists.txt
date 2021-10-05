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

#=--- [particle update] -----------------------------------------------------==#

add_executable(
  particle_update ${ripple_SOURCE_DIR}/benchmarks/particle/particle_update.cu)
target_link_libraries(particle_update ripple::ripple -latomic)

if (RIPPLE_BUILD_WITH_KOKKOS)
  add_executable(
    particle_update_kokkos_mpi
    ${ripple_SOURCE_DIR}/benchmarks/particle/particle_update_kokkos_mpi.cpp)
  target_link_libraries(
    particle_update_kokkos_mpi
    Kokkos::kokkos 
    -lcudart
    ${MPI_CXX_LIBRARIES})

    add_executable(
      particle_update_kokkos_thread
      ${ripple_SOURCE_DIR}/benchmarks/particle/particle_update_kokkos_threads.cpp)
    target_link_libraries(
      particle_update_kokkos_thread
      Kokkos::kokkos 
      -lcudart
      -pthread)
endif()

#=--- [reinitialization] ----------------------------------------------------==#

add_executable(
  reinitialization 
  ${ripple_SOURCE_DIR}/benchmarks/reinitialization/reinitialization.cu)
target_link_libraries(reinitialization ripple::ripple -latomic)

#==--- [saxpy ---------------------------------------------------------------==#

add_executable(
  saxpy_cublas ${ripple_SOURCE_DIR}/benchmarks/saxpy/saxpy_cublas.cu)
target_link_libraries(saxpy_cublas ripple::ripple -lcublas)

add_executable(
  saxpy_ripple ${ripple_SOURCE_DIR}/benchmarks/saxpy/saxpy_ripple.cu)
target_link_libraries(saxpy_ripple ripple::ripple -latomic)
#==--- ripple/benchmarks/CMakeLists.txt -------------------------------------==#
#
#                      Copyright (c) 2019 Rob Clucas
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

#==--- [force] --------------------------------------------------------------==#

add_executable(
  force ${ripple_SOURCE_DIR}/benchmarks/force/force.cu)
target_link_libraries(force ripple::ripple)

#=--- [particle update] -----------------------------------------------------==#

add_executable(
  particle_update ${ripple_SOURCE_DIR}/benchmarks/particle/particle_update.cu)
target_link_libraries(particle_update ripple::ripple -latomic)

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
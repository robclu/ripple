#==--- ripple/apps/CMakeLists.txt -------------------------------------------==#
#
#                      Copyright (c) 2019, 2020 Rob Clucas
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#


add_executable(numa_arch numa_arch.cpp)
target_link_libraries(
  numa_arch ripple::ripple -lcudart_static -ldl -lrt -lcudadevrt -pthread)
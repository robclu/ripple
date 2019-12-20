//==--- ripple/apps/numa_arch.cpp -------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file numa_arch.cpp
/// \brief This file defines an application to determine and print the numa
///        architecture.
//
//==------------------------------------------------------------------------==//

#include <ripple/arch/cpuinfo.hpp>
#include <ripple/arch/topology.hpp>
#include <iostream>

int main(int argc, char** argv) {
  auto cpu_info = ripple::CpuInfo();
  auto topology = ripple::Topology();

  printf("Available cores   : %-6i\n" , cpu_info.available_cores());
  printf("Packages          : %-6i\n" , cpu_info.packages());
  printf("Cores per package : %-6i\n" , cpu_info.cores_per_package());
  printf("Available gpus    : %-6zu\n", topology.num_gpus());
  printf("Total gpu mem (GB): %-6lu\n", topology.combined_gpu_memory_gb());

  printf("Cache information:\n");
  for (auto& cache : cpu_info.cache_info()) {
    printf("{\n"
      "\tLevel           : %-6i\n"
      "\tType            : %-6i\n"
      "\tLinesize        : %-6i\n"
      "\tPartitions      : %-6i\n"
      "\tAssosciativity  : %-6i\n"
      "\tMax proc shared : %-6i\n"
      "\tSets            : %-6i\n"
      "\tSize (kB)       : %-6i\n"
      "}\n",
      cache.level         ,
      cache.type          ,
      cache.linesize      ,
      cache.partitions    ,
      cache.assosciativity,
      cache.shared_by     ,
      cache.sets          ,
      cache.size()
    );
  }

  printf("CPU indices (Package,Core,Thread):\n{\n");
  for (auto& proc : cpu_info.processor_info()) {
    printf("\t{%2i %2i %2i }\n", proc.package, proc.core, proc.thread);
  }
  printf("\n}\n");

  printf("GPU Peer access: {Device id : [peer ids]}:\n{\n");
  for (auto & gpu : topology.gpus) {
    printf("\t{%2i : [", gpu.index);
    for (auto& peer : gpu.peers) {
      printf("%2i ", peer);
    }
    printf("] }");
  }
  printf("\n}\n");
}


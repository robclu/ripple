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
#include <iostream>

int main(int argc, char** argv) {
  auto cpu_info = ripple::CpuInfo();

  printf("Available cores   : %-6i\n", cpu_info.available_cores());
  printf("Packages          : %-6i\n", cpu_info.packages());
  printf("Cores per package : %-6i\n", cpu_info.cores_per_package());

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
    printf("\t{%2i %2i %2i}\n", proc.package, proc.core, proc.thread);
  }
  printf("\n}\n");
}


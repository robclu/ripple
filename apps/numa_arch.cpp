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
  auto caches   = cpu_info.create_cache_info();

  for (auto& cache : caches) {
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
}


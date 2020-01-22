//==--- ripple/arch/topology.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  topology.hpp
/// \brief This file defines a struct to store the topology of the system.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_TOPOLOGY_HPP
#define RIPPLE_ARCH_TOPOLOGY_HPP

#include "gpu_info.hpp"

namespace ripple {

/// The Topology struct stores information about the topology of the system. It
/// holds information about the caches, processors, processor mapping, gpu
/// devies, etc, which can be used to optimally partiion parallel algorithms.
struct Topology {
  /// Constructor to create the topology.
  Topology() {
    gpus = GpuInfo::create_for_all_devices();
  }

  /// Returns the total number of gpus in the system.
  auto num_gpus() -> size_t {
    return gpus.size();
  }

  /// Returns the total number of bytes of memory available from all gpus.
  auto combined_gpu_memory() const -> uint64_t {
    uint64_t total_mem = 0;
    for (const auto& gpu : gpus) {
      total_mem += gpu.mem_size;
    }
    return total_mem;
  } 

  /// Returns the total number of gb of available memory from all gpus.
  auto combined_gpu_memory_gb() const -> float {
    return static_cast<float>(combined_gpu_memory()) / float{1073741824};
  }

  std::vector<GpuInfo> gpus;      //!< Gpus for the system.
  CpuInfo              cpu_info;  //!< Cpu info for the system.
};

} // namespace ripple

#endif // RIPPLE_ARCH_TOPOLOGY_HPP

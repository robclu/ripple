/**==--- ripple/core/arch/topology.hpp --------------------- -*- C++ -*- ---==**
 *
 *                                Ripple
 *
 *                 Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==------------------------------------------------------------------------==**
 * \file  topology.hpp
 * \brief This file defines a struct to store the topology of the system.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ARCH_TOPOLOGY_HPP
#define RIPPLE_ARCH_TOPOLOGY_HPP

#include "cpu_info.hpp"
#include "gpu_info.hpp"
#include <map>

namespace ripple {

/**
 * Defines a map type to map gpu indices to stream indices.
 */
using StreamMap = std::map<ripple::GpuInfo::Index, ripple::GpuInfo::Id>;

/**
 * The Topology struct stores information about the topology of the system. It
 * holds information about the caches, processors, processor mapping, gpu
 * devies, etc, which can be used to optimally partiion parallel algorithms.
 */
struct Topology {
 private:
  /** Conversion factor for bytes to gb. */
  static constexpr float bytes_to_gb = 1.0f / 1073741824.0f;

 public:
  /**
   * Constructor to create the topology.
   */
  Topology() noexcept {
    gpus = GpuInfo::create_for_all_devices();
  }

  /**
   * Gets the number of gpus in the system.
   * \return The total number of gpus in the system.
   */
  auto num_gpus() const noexcept -> size_t {
    return gpus.size();
  }

  /**
   * Gets the number of cpu cores for the system.
   * \return The number of independent cpu cores in the system.
   */
  auto num_cores() const noexcept -> size_t {
    return cpu_info.available_cores();
  }

  /**
   * Calculates the total number of bytes of memory available from all gpus.
   * \return The total number of bytes of gpu memory in the system.
   */
  auto combined_gpu_memory() const noexcept -> uint64_t {
    uint64_t total_mem = 0;
    for (const auto& gpu : gpus) {
      total_mem += gpu.mem_size;
    }
    return total_mem;
  }

  /**
   * Calculates the total number of gigabytes of memory available from all gpus.
   * \return The total number of gigabytes of gpu memory in the system.
   */
  auto combined_gpu_memory_gb() const noexcept -> float {
    return static_cast<float>(combined_gpu_memory()) * bytes_to_gb;
  }

  /**
   * Determines if peer to peer access is possible between two gpus.
   * \param size_t gpu_id1 The index of the first gpu.
   * \param size_t gpu_id2 The index of the second gpu.
   * \return true if peer to peer is possible between the gpus.
   */
  auto device_to_device_available(size_t gpu_id1, size_t gpu_id2) const noexcept
    -> bool {
    return gpu_id1 == gpu_id2 || gpus[gpu_id1].peer_to_peer_available(gpu_id2);
  }

  std::vector<GpuInfo> gpus;     //!< Gpus for the system.
  CpuInfo              cpu_info; //!< Cpu info for the system.
};

/**
 * Gets a reference to the system topology.
 * \return A reference to the system topology.
 */
static inline auto topology() noexcept -> Topology& {
  static Topology topo;
  return topo;
}

} // namespace ripple

#endif // RIPPLE_ARCH_TOPOLOGY_HPP

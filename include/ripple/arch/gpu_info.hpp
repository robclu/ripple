//==--- ripple/arch/gpu_info.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  gpu_info.hpp
/// \brief This file defines a struct to store gpu information.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_GPU_INFO_HPP
#define RIPPLE_ARCH_GPU_INFO_HPP

#include <ripple/storage/storage_traits.hpp>
#include <ripple/utility/range.hpp>
#include <ripple/utility/portability.hpp>
#include <array>
#include <vector>

namespace ripple {

/// The GpuInfo struct stores information about the gpu.
struct GpuInfo {
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type used for the device index.
  using index_t            = uint32_t;
  /// Defines the type of container used to store peer access indices.
  using peer_container_t   = std::vector<index_t>;
  /// Defines the type of the stream for the device.
  using stream_t           = cudaStream_t;
  /// Defines the type of container for streams for the device.
  using stream_container_t = std::array<stream_t, 8>;

  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines an invalid value for processor information.
  static constexpr index_t invalid = 0xFFFFFFFF;

  /// Defines the amount of padding to avoid false sharing.
  static constexpr size_t padding_size = false_sharing_size - ((
    sizeof(peer_container_t) + 
    sizeof(index_t)          + 
    2 * sizeof(uint64_t)     +
    sizeof(stream_container_t)
  ) % false_sharing_size);

  //==--- [constructor] ----------------------------------------------------==//
  
  /// Constructor to initialise the info with a specific index.
  GpuInfo(index_t idx) : index(idx) {
    for (auto& stream : streams) {
      cudaStreamCreate(&stream);
    }
  }

  /// Destructor which cleans up the streams.
  ~GpuInfo() {
    // Currently causing a segfault, so we can't clean up the streams ...
    //for (auto& stream : streams) {
    //   cudaStreamDestroy(stream);
    //}
  }

  //==--- [interface] ------------------------------------------------------==//
  
  /// Creates the information for all gpus in the system.
  static auto create_for_all_devices() -> std::vector<GpuInfo> {
    index_t num_devices  = device_count();
    auto    devices      = std::vector<GpuInfo>();

    cudaDeviceProp device_props;
    int can_access_peer;
    for (auto dev : range(num_devices)) {
      auto& info = devices.emplace_back(dev);
      cudaSetDevice(dev);
      cudaGetDeviceProperties(&device_props, dev);

      info.mem_size = device_props.totalGlobalMem;

      // Determine if peer to peer is supported:
      for (index_t i = 0; i < num_devices; ++i) {
        if (i == dev) {
          continue;
        }
        cudaDeviceCanAccessPeer(&can_access_peer, dev, i);
        if (can_access_peer) {
          info.peers.emplace_back(i);
        }
      }
    }
    return devices;
  }
  
  /// Returns the total number of gpus detected in the system.
  static auto device_count() -> uint32_t {
    int count;
    cudaGetDeviceCount(&count);
    return static_cast<uint32_t>(count);
  }

  /// Returns true if any field in the information is invalid.
  auto is_invalid() const -> bool {
    return index == invalid;
  }

  /// Returns if the gpu has any peer to peer access with another device.
  auto peer_to_peer_available() const -> bool {
    return !peers.empty();
  }

  /// Returns if the gpu can access peer memory with the device with index \p
  /// other_id.
  /// \param other_id The index of teh other gpu to check peer access with.
  auto peer_to_peer_available(index_t other_id) -> bool {
    for (auto& peer_id : peers) {
      if (peer_id == other_id) {
        return true;
      }
    }
    return false;
  }

  /// Returns the amount of memory which is unallocated on the gpu.
  auto mem_remaining() const -> uint64_t {
    return mem_alloc < mem_size ? mem_size - mem_alloc : 0;
  }

  stream_container_t streams;              //!< Streams for the device.
  peer_container_t   peers;                //!< Default to no peers.
  index_t            index     = invalid;  //!< Index of the gpu in the system.
  uint64_t           mem_size  = 0;        //!< Amount of memory for the device.
  uint64_t           mem_alloc = 0;        //!< Amount of memory alocated.     
  uint8_t            pad[padding_size];    //!< Padding for false sharing.
};

} // namespace ripple

#endif // RIPPLE_ARCH_GPU_INFO_HPP

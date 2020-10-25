//==--- ripple/core/arch/gpu_info.hpp ---------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019 Rob Clucas
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

#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/core/utility/range.hpp>
#include <ripple/core/utility/portability.hpp>
#include <array>
#include <vector>

namespace ripple {

/**
 * The GpuInfo struct stores information about the gpu.
 */
struct GpuInfo {
  /** The number of streams for the cpu. */
  static constexpr size_t streams_per_device = 2;

  /*==--- [aliases] --------------------------------------------------------==*/

  // clang-format off
  /** Type used for the device index. */
  using index_t            = uint32_t;
  /** Type of container used to store peer access indices. */
  using peer_container_t   = std::vector<index_t>;
  /** Type of the stream for the device. */
  using stream_t           = cudaStream_t;
  /** Type of container for streams for the device. */
  using stream_container_t = std::array<stream_t, streams_per_device>;
  // clang-format on

  /*==--- [constants] ------------------------------------------------------==*/

  /** Value for invalid processor information. */
  static constexpr index_t invalid = 0xFFFFFFFF;

  // clang-format off
  /** The amount of padding to avoid false sharing. */
  static constexpr size_t padding_size = avoid_false_sharing_size - ((
    sizeof(peer_container_t) + 
    sizeof(index_t)          + 
    sizeof(uint64_t)         +
    sizeof(uint64_t)         +
    sizeof(stream_container_t)
  ) % avoid_false_sharing_size);
  // clang-format on

  /*==--- [constructor] ----------------------------------------------------==*/

  /**
   * Constructor to initialise the info with a specific index.
   * \param idx The index of the gpu for the info.
   */
  GpuInfo(index_t idx) noexcept : index{idx}, peers{idx} {
    cudaSetDevice(index);
    for (auto& stream : streams) {
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }
  }

  /**
   * Destructor which cleans up the streams.
   */
  ~GpuInfo() noexcept {
    // Currently causing a segfault, so we can't clean up the streams ...
    // cudaSetDevice(index);
    // for (auto& stream : streams) {
    //  cudaStreamDestroy(stream);
    //}
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Creates the information for all gpus in the system.
   * \return A vector of information for all the system gpus.
   */
  static auto create_for_all_devices() -> std::vector<GpuInfo> {
    index_t num_devices = device_count();
    auto    devices     = std::vector<GpuInfo>();

    cudaDeviceProp device_props;
    int            can_access_peer;
    for (auto dev : range(num_devices)) {
      // Constructor sets the device to the current device.
      cudaSetDevice(dev);
      auto& info = devices.emplace_back(dev);
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
          cudaDeviceEnablePeerAccess(i, 0);
        }
      }
    }
    return devices;
  }

  /**
   * Gets the total number of gpus detected in the system.
   * \return The number of devices in the system.
   */
  static auto device_count() noexcept -> uint32_t {
    int count;
    cudaGetDeviceCount(&count);
    return static_cast<uint32_t>(count);
  }

  /**
   * Determines if the info is invalid.
   * \return true if the index matches the invalid index.
   */
  auto is_invalid() const noexcept -> bool {
    return index == invalid;
  }

  /**
   * Determines if the gpu has any peer to peer access with another device.
   * \return true if the gpu can directly access any other gpu's memory.
   */
  auto peer_to_peer_available() const noexcept -> bool {
    return !peers.empty();
  }

  /**R
   * Determines if this gpu can directly access the memory of the gpu with the
   * given index.
   * \param other_id The index of the other gpu to check peer access with.
   * \return true if this gpu can directly access the other gpu's memory.
   */
  auto peer_to_peer_available(index_t other_id) const noexcept -> bool {
    for (auto& peer_id : peers) {
      if (peer_id == other_id) {
        return true;
      }
    }
    return false;
  }

  /**
   * Gets the amount of memory which is unallocated on the gpu.
   *
   * \note The accuracy of this depends on the applications tracking of memory
   *       allocations.
   *
   * \return The amount of memory remaining for the gpu.
   */
  auto mem_remaining() const noexcept -> uint64_t {
    return mem_alloc < mem_size ? mem_size - mem_alloc : 0;
  }

  /**
   * Synchronizes the gpu.
   */
  auto synchronize() const noexcept -> void {
    if (index == invalid) {
      return;
    }

    cudaSetDevice(index);
    cudaDeviceSynchronize();
  }

  stream_container_t streams   = {};      //!< Streams for the device.
  peer_container_t   peers     = {};      //!< Default to no peers.
  index_t            index     = invalid; //!< Index of the gpu in the system.
  uint64_t           mem_size  = 0;       //!< Amount of memory for the device.
  uint64_t           mem_alloc = 0;       //!< Amount of device  memory.
  uint8_t            pad[padding_size];   //!< Padding for false sharing.
};

} // namespace ripple

#endif // RIPPLE_ARCH_GPU_INFO_HPP

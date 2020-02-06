//==--- ripple/core/container/block.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                  Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block.hpp
/// \brief This file defines a class for a block which has both host and device
///        data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_BLOCK_HPP
#define RIPPLE_CONTAINER_BLOCK_HPP

#include "block_traits.hpp"

namespace ripple {

/// Defines a type which wraps a host and device block into a single type, and
/// which has a state for the different data components in the block.
///
/// The Block type stores both host and device block data, and provides
/// operations to move the data between the host and device memory spaces.
///
/// \tparam T     The type of the data for the block.
/// \tparam Dims  The number of dimensions for the block.
struct Block {
  /// Defines the type of the type for the block index.
  using index_t        = size_t[Dims];
  /// Defines the type of the host block for the grid.
  using host_block_t   = HostBlock<T, Dimensions>;
  /// Defines the type of the device block for the grid.
  using device_block_t = DeviceBlock<T, Dimensions>;

  /// Defines the state of the block data.
  enum class State : uint8_t {
    invalid          = 0,   //!< Data is not valid.
    updated_host     = 1,   //!< Data is on the host and is updated.
    submitted_host   = 2,   //!< Data has been submitted on the host.
    updated_device   = 3,   //!< Data is on the device and is updated.
    submitted_device = 4    //!< Data has been submitted on the device.
  };

  //==--- [interface] ------------------------------------------------------==//
  
  /// Ensures that the data is available on the device. This will make a copy
  /// from the host to the device if the data is not already on the device.
  auto ensure_device_data_available() -> void {
    switch (data_state) {
      // Already on the device, just return:
      case State::updated_device:
        return;
      // Need to wait for whatever is happening on the host, nothing to do
      // currently ...
      case State::submitted_host:
        return;
      // On the host, so copy to the device:
      case State::updated_host: {
        cudaSetDevice(gpu_id);
        device_data = host_data;
        break;
      }
      // Work submitted to the device, just need a synchronization to ensure
      // that the operation is finished.
      case State::submitted_device:
        break;
      default:
        break;
    }
    cudaStreamSynchronize(device_data.stream());
  }

  /// Ensures that the data is available on the host. This will make a copy
  /// from the device to the host if the data is not already on the host.
  auto ensure_host_data_available() -> void {
    switch (data_state) {
      // Already on the host, just return:
      case State::updated_host:
        return;
      // Regardless of what is happening on the device, we need to add the copy
      // operation to the device's work queue, and then synchronize the stream
      // after.
      case State::submitted_device:
      case State::updated_device:
        cudaSetDevice(gpu_id);
        host_data = device_data;
        cudaStreamSynchronize(device_data.stream());
        return;
      // Work has been submitted to the host, so data is already on the host.
      case State::submitted_device:
        break;
      default:
        break;
    }
  }

  /// Ensures that the padding is available on the device. This has to copy the
  /// padding from the surrounding (2,4,8) blocks in (1D,2D,3D). This will just
  /// push the operations onto the gpuw work queue and then synchronize at the
  /// end.
  auto ensure_device_padding_available() -> void {
    // Currently just do nothing... padding not yet supported ...
  }

  /// Ensures that the padding is available on the device. This has to copy the
  /// padding from the surrounding (2,4,8) blocks in (1D,2D,3D). This will just
  /// push the operations onto the gpuw work queue and then synchronize at the
  /// end.
  auto ensure_host_padding_available() -> void {
    // Currently just do nothing... padding not yet supported ...
  }

  /// Returns true if the block has padding, otherwise returns false.
  auto has_padding() const -> bool {
    return host_data.padding() > 0;
  }

  //==--- [members] ------------------------------------------------------==//

  host_block_t   host_data;                      //!< Host block data.
  device_block_t device_data;                    //!< Device block data.
  index_t        indices;                        //!< Indices of the block.
  Block*         sibling       = nullptr;        //!< Data sharer.
  int            gpu_id        = -1;             //!< Device index.
  State          data_state    = State::invalid; //!< Data state.
  State          padding_state = State::invalid; //!< Padding state.
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BLOCK_HPP


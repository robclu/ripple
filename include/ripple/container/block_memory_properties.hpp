/**=--- ripple/container/block_memory_propertires.hpp ------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  block_memory_properties.hpp
 * \brief This file defines memory properties for blocks.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_CONTAINER_BLOCK_MEMORY_PROPERTIES_HPP
#define RIPPLE_CONTAINER_BLOCK_MEMORY_PROPERTIES_HPP

#include <cstdint>

namespace ripple {

/**
 * The BlockMemoryProps struct defines properties for the memory for a block
 * which can be used by clases to perform allocation and transfer in different
 * ways.
 *
 * By default, the properties of the memory are the following:
 *  - Not page locked
 *  - Unallocated
 *  - Asynchronous transfer
 *  - Must not free the memory
 */
struct BlockMemoryProps {
  /** Value to reset the memory properties. */
  static constexpr uint8_t reset_value = 5;

  /**
   * Default constructor for memory properties.
   */
  constexpr BlockMemoryProps() noexcept
  : pinned(1), allocated(0), async_copy(0), must_free(0) {}

  /**
   * Resets the memory properties for the block to be the following:
   *  - Not page locked
   *  - Unallocated
   *  - Synchronous transfer
   *  - Must free the memory
   */
  auto reset() noexcept -> void {
    *reinterpret_cast<uint8_t*>(this) = reset_value;
  }

  // clang-format off
  /** If the memory has been pinned (page locked). */
  uint8_t pinned     : 1;
  /** If the memory memory has been allocated. */
  uint8_t allocated  : 1;
  /** If the memory should be copied asynchronously. */
  uint8_t async_copy : 1;
  /** If the memory must be freed on destruction. */
  uint8_t must_free  : 1;
  // clang-format on
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BLOCK_MEMORY_PROPERTIES_HPP

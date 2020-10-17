//==--- ripple/core/utility/memory.hpp --------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  memory.hpp
/// \brief This file defines a utility functions for memory related operations.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_MEMORY_HPP
#define RIPPLE_UTILITY_MEMORY_HPP

#include "portability.hpp"
#include <cassert>
#include <cstdint>

namespace ripple {

/**
 * Gets a new ptr offset by the given amount from the ptr.
 *
 * \note This does __not__ ensure alignemt. If the pointer needs to be aligned,
 *       then pass the result to `align()`.
 *
 * \sa align
 *
 * \param ptr    The pointer to offset.
 * \param amount The amount to offset ptr by.
 * \return A new pointer at the offset location.
 */
ripple_host_device static inline auto
offset_ptr(const void* ptr, uint32_t amount) noexcept -> void* {
  return reinterpret_cast<void*>(uintptr_t(ptr) + amount);
}

/**
 * Gets a pointer with an address aligned to the goven alignment.
 *
 * \note In debug, this will assert at runtime if the alignemnt is not a power
 *       of two, in release, the behaviour is undefined.
 *
 * \param ptr       The pointer to align.
 * \param alignment The alignment to ensure.
 */
ripple_host_device static inline auto
align_ptr(const void* ptr, size_t alignment) noexcept -> void* {
  assert(
    !(alignment & (alignment - 1)) &&
    "Alignment must be a power of two for linear allocation!");
  return reinterpret_cast<void*>(
    (uintptr_t(ptr) + alignment - 1) & ~(alignment - 1));
}

} // namespace ripple

#endif // RIPPLE_UTILITY_MEMORY_HPP
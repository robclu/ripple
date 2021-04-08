/**=--- ripple/utility/bitops.hpp -------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  bitops.hpp
 * \brief This file contains bit related operations.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_UTILITY_BITOPS_HPP
#define RIPPLE_UTILITY_BITOPS_HPP

#include "portability.hpp"

namespace ripple {

/**
 * Gets a mask of the giver width (in bits).
 * \param width The width of the bitmask.
 * \return A mask for the given number of bits.
 */
ripple_all static inline auto bitmask(int width) -> uint64_t {
  return (uint64_t{1} << width) - 1;
}

/**
 * Gets the value of the bits between [start, end] (inclusive).
 * \param  val The value to get a bitrange from.
 * \param  start The index of the start bit.
 * \param  end   The index of the end bit.
 * \tparam T     The type of the input value.
 * \return The value of the bits between the given [start, end] range.
 */
template <typename T>
ripple_all static inline auto bits(T val, int start, int end) -> T {
  return static_cast<T>((val >> start) & bitmask(end - start + 1));
}

} // namespace ripple

#endif // RIPPLE_UTILITY_BITOPS_HPP

//==--- ripple/arch/cpuidpp ------------------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cpuid.hpp
/// \brief This file performs a cpuid instruction.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_BITOPS_HPP
#define RIPPLE_UTILITY_BITOPS_HPP

#include <cstdint>

namespace ripple {

/// Returns a mask of \p width.
/// \param width The width of the bitmask.
static inline auto bitmask(int width) -> uint64_t {
  return (uint64_t{1} << width) - 1;
}

/// Returns the value of the bits between [start, end] (inclusive).
/// \param  val The value to get a bitrange from.
/// \param  start The index of the start bit.
/// \param  end   The index of the end bit.
/// \tparma T     The type of the input value.
template <typename T>
static inline auto bits(T val, int start, int end) {
  return static_cast<T>((val >> start) & bitmask(end - start + 1));
}

} // namespace ripple

#endif // RIPPLE_UTILITY_BITOPS_HPP

//==--- ripple/math/math.hpp ------------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  math.hpp
/// \brief This file defines math functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MATH_MATH_HPP
#define RIPPLE_MATH_MATH_HPP

#include <ripple/utility/portability.hpp>
#include <math.h>

namespace ripple::math {
namespace detail {

//==--- [sign] -------------------------------------------------------------==//

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Specialization for a false (unsigned) type.
/// \param  x The value to get the sign of.
/// \tparam T The type of the data.
template <typename T>
ripple_host_device constexpr auto sign(T x, std::false_type) -> T {
  return T(0) < x;
}

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Specialization for a true (signed) type.
/// \param  x The value to get the sign of.
/// \tparam T The type of the data.
template <typename T>
ripple_host_device constexpr auto sign(T x, std::true_type) -> T {
  return (T(0) < x) - (x < T(0));
}

} // namespace detail

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0.
/// \param  x  The value to get the sign of.
/// \tparam T  The type of the data.
template <typename T>
ripple_host_device constexpr auto sign(T x) -> T {
  return detail::sign(x, std::is_signed<T>());
}

} // namespace ripple::math

#endif // RIPPLE_MATH_MATH_HPP

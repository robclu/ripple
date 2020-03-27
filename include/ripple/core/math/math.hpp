//==--- ripple/core/math/math.hpp ------------------------------- -*- C++ -*- ---==//
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

#include <ripple/core/container/array.hpp>
#include <ripple/core/utility/portability.hpp>
#include <math.h>

namespace ripple::math {

//==--- [hash] -------------------------------------------------------------==//

/// Constexpr hash function to compute the hash of the \p input.
/// \param input The input of to compute the hash of.
ripple_host_device constexpr auto hash(char const* input) -> unsigned int {
  return *input 
    ? static_cast<unsigned int>(*input) + 33 * hash(input + 1)
    : 5381;
}

/// Literal operator to perform a hash on string literals.
/// \tparam input The input string to hash.
ripple_host_device constexpr auto operator "" _hash(
  const char* input, unsigned long
)  -> unsigned int {
  return hash(input);
}

/// Computes the square root of the \p v value.
/// \param  v The value to compute the square root of.
/// \tparam T The type of the data.
template <typename T, non_array_enable_t<T> = 0>
ripple_host_device constexpr auto sqrt(const T& v) -> T {
  return std::sqrt(v);
}

/// Overload of sqrt function for array implementations.
template <typename Impl, array_enable_t<Impl> = 0>
ripple_host_device constexpr auto sqrt(const Array<Impl>& arr) -> Impl {
  auto r = Impl();
  unrolled_for<array_traits_t<Impl>::size>([&] (auto _i) {
    constexpr auto i = size_t{_i};
    r[i] = sqrt(arr[i]);
  });
  return r;
}

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

/// Returns the absolute value of \p x.
/// \param x The value to get the absolute value of.
/// \tpram T The type of the value.
template <typename T>
ripple_host_device constexpr auto abs(T x) -> T {
  return sign(x) * x;
}

} // namespace ripple::math

#endif // RIPPLE_MATH_MATH_HPP

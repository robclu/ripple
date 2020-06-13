//==--- ripple/core/math/math.hpp ------------------------------- -*- C++ -*-
//---==//
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
  return *input ? static_cast<unsigned int>(*input) + 33 * hash(input + 1)
                : 5381;
}

/// Literal operator to perform a hash on string literals.
/// \tparam input The input string to hash.
ripple_host_device constexpr auto
operator"" _hash(const char* input, unsigned long) -> unsigned int {
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
/// \param  arr  The array to compute the square root for each element.
/// \tparam Impl The implementation type of the array interface.
template <typename Impl, array_enable_t<Impl> = 0>
ripple_host_device constexpr auto sqrt(const Array<Impl>& arr) -> Impl {
  auto r = Impl();
  unrolled_for<array_traits_t<Impl>::size>([&](auto _i) {
    constexpr auto i = size_t{_i};
    r[i]             = sqrt(arr[i]);
  });
  return r;
}

namespace detail {

//==--- [xorshift] ---------------------------------------------------------==//

/// Implementation of a random number generator using the xor32 method, as
/// described here:
///
///   [xorshift32](https://en.wikipedia.org/wiki/Xorshift)
///
/// Returns a random 32 bit integer in the range [0, 2^32 - 1].
ripple_host_device static inline auto xorshift_32() noexcept -> uint32_t {
  static uint32_t rand_seed = 123456789;
  uint32_t        x         = rand_seed;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rand_seed = x;
  return x;
}

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

//==--- [log base 2] -------------------------------------------------------==//

/// Computes the base 2 log of a 32 bit integer.
/// \param value The value to find log base 2 of.
ripple_host_device constexpr auto log_2(uint32_t value) noexcept -> uint32_t {
  // clang-format off
  uint32_t result = 0, shift = 0;
  result = (value > 0xFFFF) << 4; value >>= result;
  shift  = (value > 0xFF  ) << 3; value >>= shift ; result |= shift;
  shift  = (value > 0xF   ) << 2; value >>= shift ; result |= shift;
  shift  = (value > 0x3   ) << 1; value >>= shift ; result |= shift;
  return result |= (value >> 1);
  // clang-format on
}

//==--- [randint] ----------------------------------------------------------==//

/// Returns a random 32 bit integer in the range [\p start, \p end]. It's
/// very fast (approx ~1-2ns based on benchmarks, in comparison
/// std::experimental::randint, which is around 16ns), however, the
/// implementation is known to fail some of the random number generation tests.
/// It also only provides a uniform distribution. It should be used only when
/// performance is critical and the quality of the generated random number is
/// not hugely important.
///
/// \param start The starting value of the range.
/// \param end   The end value of the range.
/// Returns a uniformly distributed random number between
ripple_host_device static inline auto
randint(uint32_t start, uint32_t end) noexcept -> uint32_t {
  const uint32_t range = end - start;
  return (detail::xorshift_32() >> (32 - log_2(range) - 1)) % range + start;
}

//==--- [sign] -------------------------------------------------------------==//

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0.
/// \param  x  The value to get the sign of.
/// \tparam T  The type of the data.
template <typename T>
ripple_host_device constexpr auto sign(T x) -> T {
  return detail::sign(x, std::is_signed<T>());
}

//==--- [abs] --------------------------------------------------------------==//

/// Returns the absolute value of \p x.
/// \param  x The value to get the absolute value of.
/// \tparam T The type of the value.
template <typename T, non_array_enable_t<T> = 0>
ripple_host_device constexpr auto abs(T x) -> T {
  return sign(x) * x;
}

/// Overload of sqrt function for array implementations.
/// \param  arr  The array to compute the square root for each element.
/// \tparam Impl The implementation type of the array interface.
template <typename Impl, array_enable_t<Impl> = 0>
ripple_host_device constexpr auto abs(const Array<Impl>& arr) -> Impl {
  auto r = Impl();
  unrolled_for<array_traits_t<Impl>::size>([&](auto _i) {
    constexpr auto i = size_t{_i};
    r[i]             = abs(arr[i]);
  });
  return r;
}

//==--- [isnan] ------------------------------------------------------------==//

/// Returns true if any of the elements in the array \p a ara nan, othterwise
/// returns false.
/// \param[in] a  The array to check for nan elements.
/// \tparam    T  The type of the data.
/// \tparam    S  The number of elements in the arrays.
template <typename Impl>
ripple_host_device constexpr auto isnan(const Array<Impl>& a) -> bool {
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i])) {
      return true;
    }
  }
  return false;
}

/// Returns true if any of the elements in the array \p a ara nan, othterwise
/// returns false. This is a wrapper around std::isnan so that it works with
/// arrays as well as build in numerical types.
/// \param[in] a  The data to check for nan.
/// \tparam    T  The type of the data.
/// \tparam    S  The number of elements in the arrays.
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
ripple_host_device constexpr auto isnan(T a) -> bool {
  return std::isnan(a);
}

//==--- [dot] --------------------------------------------------------------==//

template <typename ImplA, typename ImplB>
ripple_host_device constexpr auto
dot(const Array<ImplA>& a, const Array<ImplB>& b) ->
  typename array_traits_t<ImplA>::value_t {
  using traits_t = array_traits_t<ImplA>;
  using value_t  = typename traits_t::value_t;
  value_t r      = 0;
  unrolled_for_bounded<traits_t::size>([&](auto i) {
    r += static_cast<value_t>(a[i]) * static_cast<value_t>(b[i]);
  });
  return r;
}

} // namespace ripple::math

#endif // RIPPLE_MATH_MATH_HPP

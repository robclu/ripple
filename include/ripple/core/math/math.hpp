//==--- ripple/core/math/math.hpp -------------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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
#include <algorithm>
#include <cmath>

namespace ripple::math {

/*==--- [hash] -------------------------------------------------------------==*/

/**
 * Constexpr hash function to compute the hash of the \p input.
 *
 * \note This con be used with compile time values, and can be used with
 *       `_hash` in switch statements.
 *
 * \param input The input of to compute the hash of.
 * \return The hash of the input.
 */
ripple_host_device constexpr auto
hash(char const* input) noexcept -> unsigned int {
  return *input ? static_cast<unsigned int>(*input) + 33 * hash(input + 1)
                : 5381;
}

/**
 * Constexpr hash of two 32 bit unsigned ints. This uses the cantour pairing
 * formula and is composible, i.e it can be used twice for 3 ints, etc.
 *
 * \note : It is possible that this overflows for large values of \p a and \p
 *         b, so test for different use cases.
 *
 * \param a The first int for the hash.
 * \param b The second input for the hash.
 * \return The combined hash of the two inputs.
 */
ripple_host_device constexpr auto
hash_combine(uint32_t a, uint32_t b) noexcept -> uint64_t {
  constexpr uint64_t div_factor = 2;
  uint64_t           x = a, y = b;
  return (x + y) * (x + y + uint64_t{1}) / div_factor + y;
}

/**
 * Literal operator to perform a hash on string literals.
 * \tparam input The input string to hash.
 */
ripple_host_device constexpr auto
operator"" _hash(const char* input, unsigned long) noexcept -> unsigned int {
  return hash(input);
}

namespace detail {

/*==--- [xorshift] ---------------------------------------------------------==*/

/**
 * Implementation of a random number generator using the xor32 method, as
 * described here: [xorshift32](https://en.wikipedia.org/wiki/Xorshift)
 *
 * \return A random 32 bit integer in the range [0, 2^32 - 1].
 */
ripple_host_device static inline auto xorshift_32() noexcept -> uint32_t {
  static uint32_t rand_seed = 123456789;
  uint32_t        x         = rand_seed;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rand_seed = x;
  return x;
}

/**
 * Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
 * is greater than 0. Specialization for a false (unsigned) type.
 * \param  x The value to get the sign of.
 * \tparam T The type of the data.
 * \return The sign of the input.
 */
template <typename T>
ripple_host_device constexpr auto sign(T x, std::false_type) noexcept -> T {
  return T(0) < x;
}

/**
 * Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
 * is greater than 0. Specialization for a true (signed) type.
 * \param  x The value to get the sign of.
 * \tparam T The type of the data.
 * \return The sign of the input.
 */
template <typename T>
ripple_host_device constexpr auto sign(T x, std::true_type) noexcept -> T {
  return (T(0) < x) - (x < T(0));
}

} // namespace detail

/*==--- [general] ----------------------------------------------------------==*/

/**
 * Computes the base 2 log of a 32 bit integer.
 * \param value The value to find log base 2 of.
 * \return The log of the input.
 */
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

/**
 * Returns a random 32 bit integer in the range [\p start, \p end]. It's
 * very fast (approx ~1-2ns based on benchmarks, in comparison
 * std::experimental::randint, which is around 16ns), however, the
 * implementation is known to fail some of the random number generation tests.
 *
 * It also only provides a uniform distribution and can be used when
 * performance is critical and the quality of the generated random number is
 * not hugely important.
 *
 * \param start The starting value of the range.
 * \param end   The end value of the range.
 * \return A uniformly distributed random number in the given range.
 */
ripple_host_device static inline auto
randint(uint32_t start, uint32_t end) noexcept -> uint32_t {
  const uint32_t range = end - start;
  return (detail::xorshift_32() >> (32 - log_2(range) - 1)) % range + start;
}

/**
 * Computes the square root of the \p v value.
 *
 * \note This is provided so that calls to sqrt in this namespace will work on
 *       both array and non-array types.
 *
 * \param  v The value to compute the square root of.
 * \tparam T The type of the data.
 * \return The square root of the value.
 */
template <typename T, non_array_enable_t<T> = 0>
ripple_host_device constexpr auto sqrt(const T& v) noexcept -> T {
  return std::sqrt(v);
}

/**
 * Overload of sqrt function for array implementations.
 * \param  arr  The array to compute the square root for each element.
 * \tparam Impl The implementation type of the array interface.
 * \return A new array with all elements square rooted.
 */
template <typename Impl, array_enable_t<Impl> = 0>
ripple_host_device constexpr auto
sqrt(const Array<Impl>& arr) noexcept -> Impl {
  auto r = Impl{};
  unrolled_for<array_traits_t<Impl>::size>([&](auto _i) {
    constexpr auto i = size_t{_i};
    r[i]             = sqrt(arr[i]);
  });
  return r;
}

/**
 * Computes the sign of the input, returning -1.0 if x is less than 0.0, 0.0 if
 * x is equal to 0.0, and +1.0 if x is greater than 0.
 * \param  x  The value to get the sign of.
 * \tparam T  The type of the data.
 * \return The sign of the input.
 */
template <typename T>
ripple_host_device constexpr auto sign(T x) noexcept -> T {
  return detail::sign(x, std::is_signed<T>());
}

/**
 * Computes the absolute value of \p x.
 *
 * \note This overload is provided so that calls to abs in this namespace are
 *       valid for both array and non-array types.
 *
 * \param  x The value to get the absolute value of.
 * \tparam T The type of the value.
 */
template <typename T, non_array_enable_t<T> = 0>
ripple_host_device constexpr auto abs(T x) noexcept -> T {
  return sign(x) * x;
}

/**
 * Overload of abs function for array implementations.
 * \param  arr  The array to compute the abs value for each element.
 * \tparam Impl The implementation type of the array interface.
 * \return A new array with each element abs valued.
 */
template <typename Impl, array_enable_t<Impl> = 0>
ripple_host_device constexpr auto abs(const Array<Impl>& arr) noexcept -> Impl {
  auto r = Impl{};
  unrolled_for<array_traits_t<Impl>::size>([&](auto _i) {
    constexpr auto i = size_t{_i};
    r[i]             = abs(arr[i]);
  });
  return r;
}

/**
 * Determines if the value is nan.
 *
 * \note This overload is provided so that isnan can be used from this namespace
 *       for both array and non-array types.
 *
 * \param  a  The data to check for nan.
 * \tparam T  The type of the data.
 * \tparam S  The number of elements in the arrays.
 * \return true if the input is nan.
 */
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
ripple_host_device constexpr auto isnan(T a) noexcept -> bool {
  return std::isnan(a);
}

/**
 * Determines if any of the elements in the array \p a ara nan.
 * \param  a    The array to check for nan elements.
 * \tparam Impl The implementation type of the array interface.
 * \return true if any of the elements are nan.
 */
template <typename Impl, array_enable_t<Impl> = 0>
ripple_host_device constexpr auto isnan(const Array<Impl>& a) noexcept -> bool {
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i])) {
      return true;
    }
  }
  return false;
}

/**
 * Computes the division $\frac{num}{denom}$ in the given precision, hen
 * computes the ceil of the result before casting back to the original type.
 *
 * \param  num   The numerator for the division.
 * \param  denon The denominator for the division.
 * \tparam T     The types for the operations.
 * \tparam F     The precision to cast to.
 * \return The ceil of the division of num and denom.
 */
template <typename T, typename F = float>
ripple_host_device auto div_then_ceil(T num, T denom) noexcept -> T {
  return static_cast<T>(std::ceil(static_cast<F>(num) / denom));
}

/**
 * Computes the min of a and b.
 *
 * \note This overload is provided so that min can be used in this namespace for
 *       both array and non-array types.
 *
 * \param  a The first argument for min.
 * \param  b The second argument for min.
 * \tparam T The type of the inputs.
 * \return The min of a and b.
 */
template <typename T>
ripple_host_device constexpr auto min(const T& a, const T& b) noexcept -> T {
  return std::min(a, b);
}

/**
 * Computes the elementwise min of a and b.
 * \param  a     The first array for comparison.
 * \param  b     The second array for comparison.
 * \tparam ImplA The implementation type of the first array.
 * \tparam ImplB The implementation type of the second array.
 * \return A new array with each element as the min of the corresponding
 *         elements.
 */
template <typename ImplA, typename ImplB>
ripple_host_device constexpr auto
min(const Array<ImplA>& a, const Array<ImplB>& b) noexcept
  -> array_impl_t<ImplA, ImplB> {
  using Result = array_impl_t<ImplA, ImplB>;
  Result r;
  unrolled_for_bounded<array_traits_t<ImplA>::size>(
    [&](auto i) { r[i] = std::min(a[i], b[i]); });
  return r;
}

/**
 * Computes the max of a and b.
 *
 * \note This overload is provided so that max can be used in this namespace for
 *       both array and non-array types.
 *
 * \param  a The first argument for max.
 * \param  b The second argument for max.
 * \tparam T The type of the inputs.
 * \return The max of a and b.
 */
template <typename T>
ripple_host_device constexpr auto max(const T& a, const T& b) noexcept -> T {
  return std::max(a, b);
}

/**
 * Computes the elementwise max of a and b.
 * \param  a     The first array for comparison.
 * \param  b     The second array for comparison.
 * \tparam ImplA The implementation type of the first array.
 * \tparam ImplB The implementation type of the second array.
 * \return A new array with each element as the max of the corresponding
 *         elements.
 */
template <typename ImplA, typename ImplB>
ripple_host_device constexpr auto
max(const Array<ImplA>& a, const Array<ImplB>& b) noexcept
  -> array_impl_t<ImplA, ImplB> {
  using Result = array_impl_t<ImplA, ImplB>;
  Result r;
  unrolled_for_bounded<array_traits_t<ImplA>::size>(
    [&](auto i) { r[i] = std::max(a[i], b[i]); });
  return r;
}

/**
 * Clamps the value to the given range.
 *
 * \note This overload is provided so that clamp can be used from this namespace
 *       to both array and non-array types.
 *
 * \param  v  The value to clamp to the range.
 * \param  lo The lower value to clamp to.
 * \param  hi The higher value to clamp to.
 * \tparam T  The types of the values.
 * \return A reference to lo if v is less than lo, reference to hi if v is
 *        greater than hi, otherwise a referene to v.
 */
template <typename Impl, typename T>
ripple_host_device auto
clamp(const T& v, const T& lo, const T& hi) noexcept -> const T& {
  return std::clamp(v, lo, hi);
}

/**
 * Clamps all values in the array to the given range.
 * \param  a     The array to slamp the values for.
 * \param  lo    The lower value to clamp to.
 * \param  hi    The higher value to clamp to.
 * \tparam Impl  The implementation type of the array interface.
 * \tparam T     The type of the bounds.
 */
template <typename Impl, typename T>
ripple_host_device auto
clamp(Array<Impl>& a, const T& lo, const T& hi) noexcept -> void {
  using Value = typename array_traits_t<Impl>::value_t;
  unrolled_for<array_traits_t<Impl>::size>(
    [&](auto i) { a[i] = std::clamp(a[i], lo, hi); });
}

/*==--- [vec operations] ---------------------------------------------------==*/

/**
 * Computes the dot product of the two array types. If the arrays are small,
 * then the loop is unrolled.
 * \param  a     The first array for the dot product.
 * \param  b     The second array for the dot product.
 * \tparam ImplA The implementation type of array a.
 * \tparam ImplB The implementation type of array b.
 * \return The dot product of the arrays.
 */
template <typename ImplA, typename ImplB>
ripple_host_device constexpr auto
dot(const Array<ImplA>& a, const Array<ImplB>& b) noexcept ->
  typename array_traits_t<ImplA>::value_t {
  using traits_t = array_traits_t<ImplA>;
  using value_t  = typename traits_t::value_t;
  value_t r      = 0;
  unrolled_for_bounded<traits_t::size>([&](auto i) {
    r += static_cast<value_t>(a[i]) * static_cast<value_t>(b[i]);
  });
  return r;
}

/**
 * Computes the dot product ofthe array with itself.
 * \param  a    The array to dot product with itself.
 * \tparam Impl The implementation type of array a.
 * \return The dot product of the array with itself.
 */
template <typename Impl>
ripple_host_device constexpr auto
dot2(const Array<Impl>& a) noexcept -> typename array_traits_t<Impl>::value_t {
  using traits_t = array_traits_t<Impl>;
  using value_t  = typename traits_t::value_t;
  value_t r      = 0;
  unrolled_for_bounded<traits_t::size>([&](auto i) {
    r += static_cast<value_t>(a[i]) * static_cast<value_t>(a[i]);
  });
  return r;
}

/**
 * Cross product, this implementation is for 2d vectors, and returns the
 * magnitude that would result along the missing z direction.
 * \param  a     The first vector for the cross product.
 * \param  b     The second vector for the cross profuct.
 * \tparam ImplA The implementation type of the first array.
 * \tparam ImplB The implementation type of the second array.
 * \return The cross profuct of the two vectors.
 */
template <
  typename ImplA,
  typename ImplB,
  array_size_enable_t<ImplA, 2> = 0,
  array_size_enable_t<ImplB, 2> = 0>
ripple_host_device constexpr auto
cross(const Array<ImplA>& a, const Array<ImplB>& b) noexcept ->
  typename array_traits_t<ImplA>::value_t {
  return a[0] * b[1] - b[0] * a[1];
}

/**
 * Cross product, this implementation is for 3d vectors, and returns a
 * vector perpendicular to both.
 *
 * \param  a     The first vector for the cross product.
 * \param  b     The second vector for the cross profuct.
 * \tparam ImplA The implementation type of the first array.
 * \tparam ImplB The implementation type of the second array.
 * \return The cross profuct of the two vectors.
 */
template <
  typename ImplA,
  typename ImplB,
  array_size_enable_t<ImplA, 3> = 0,
  array_size_enable_t<ImplB, 3> = 0>
ripple_host_device constexpr auto
cross(const Array<ImplA>& a, const Array<ImplB>& b) noexcept
  -> array_impl_t<ImplA, ImplB> {
  using Result = array_impl_t<ImplA, ImplB>;
  return Result{a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]};
}

/**
 * Returns the length of the array.
 * \param  a    The array to compute the length of.
 * \tparam Impl The implementation type of the array.
 * \return The length of the array.
 */
template <typename Impl>
ripple_host_device auto length(const Array<Impl>& a) noexcept ->
  typename array_traits_t<Impl>::value_t {
  using Value = typename array_traits_t<Impl>::value_t;
  Value r{0};
  unrolled_for_bounded<array_traits_t<Impl>::size>(
    [&](auto i) { r += a[i] * a[i]; });
  return std::sqrt(r);
}

} // namespace ripple::math

#endif // RIPPLE_MATH_MATH_HPP

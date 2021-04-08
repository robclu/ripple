/**=--- ripple/container/array.hpp ------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  array.hpp
 * \brief This file defines an interface for arrays.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_CONTAINER_ARRAY_HPP
#define RIPPLE_CONTAINER_ARRAY_HPP

#include "array_traits.hpp"
#include <ripple/algorithm/unrolled_for.hpp>

namespace ripple {

/**
 * The Array class defines a static interface for array type classes. The
 * implementation is provided by the template type Impl, and any derived types
 * which don't define the necessary functions will fail at compile time when
 * the requried function is called.
 *
 * \tparam Impl The implementation of the array interface.
 */
template <typename Impl>
struct Array {
  /** The number of elements in the array. */
  static constexpr size_t elements = array_traits_t<Impl>::size;

 public:
  /**
   * Gets the value at position \p i in the array.
   *
   * \note This may return a reference or a value, depending on the
   *       implementation.
   *
   * \param i The index of the element to return.
   * \return The element at position i in the array.
   */
  ripple_all constexpr decltype(auto) operator[](size_t i) noexcept {
    return impl()->operator[](i);
  }

  /**
   * Gets the value at position \p i in the array.
   *
   * \note This may return a reference or a value, depending on the
   *       implementation.
   *
   * \param i The index of the element to return.
   */
  ripple_all constexpr decltype(auto)
  operator[](size_t i) const noexcept {
    return impl()->operator[](i);
  }

  /**
   * Gets the number of elements in the array.
   * \return The number of elements in the array.
   */
  ripple_all constexpr auto size() const noexcept -> size_t {
    return array_traits_t<Impl>::size;
  }

  /*==--- [comparison operators] -------------------------------------------==*/

  /**
   * Overload of equality operator to compare one array to another.
   * \param  other     The other array to compare against this one.
   * \tparam ImplOther The implementation type of the other array.
   * \return true if all elements of both arrays are the same, false otherwise.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator==(const Array<ImplOther>& other) noexcept -> bool {
    assert_size_match<ImplOther>();
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) != other[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of equality operator to compare all elements of the array to
   * the given value.
   *
   * \note This overload is only enabled if the type T is comparible to the type
   *       stored in the array.
   *
   * \param  val The value to compare to.
   * \tparam T   The type of the value.
   * \return true if all array elements are equal to the value.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator==(T val) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) != val) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of inequality operator to compare one array not equal to another.
   * \param  other     The other array to compare against this one.
   * \tparam ImplOther The implementation type of the other array.
   * \return true if any element of the arrays are not equal, false otherwise.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator!=(const Array<ImplOther>& other) noexcept -> bool {
    return !(*this == other);
  }

  /**
   * Overload of inequality operator to compare if any elements of the array
   * are not equal to the given value.
   * \param  val The value to compare to.
   * \tparam T   The type of the value.
   * \return true if any elements of the array are not equal to the value.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator!=(T val) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) != val) {
        return true;
      }
    }
    return false;
  }

  /**
   * Overload of less than or equal operator to compare one array to another.
   * \param  other     The other array to compare against this one.
   * \tparam ImplOther The implementation type of the other array.
   * \return true if _all_ elements of this array are <= to the corresponding
   *         elements in the other array.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator<=(const Array<ImplOther>& other) noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) > other[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of comparison operator to compare if _all_ elements of the array
   * are <= the given value.
   * \param  val The value to compare to.
   * \tparam T   The type of the value.
   * \return true if all elements are less than or equal to the given value.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator<=(T val) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) > val) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of greater than or equal operator to compare one array to
   * another.
   * \param  other     The other array to compare against this one.
   * \tparam ImplOther The implementation type of the other array.
   * \return true if _all_ elements of this array are >= the corresponding
   *         elements in the other array.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator>=(const Array<ImplOther>& other) noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) < other[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of greater than or equal operator to compare if all elements of
   * the array are greater than or equal to the given value.
   * \param  val The value to compare to.
   * \tparam T   The type of the value.
   * \return true if all elements are greater than or equal to the given value.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator>=(T val) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) < val) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of less than operator to compare one array to another.
   * \param  other     The other array to compare against this one.
   * \tparam ImplOther The implementation type of the other array.
   * \return true if all elements of this array are less than the other array
   *         elements.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator<(const Array<ImplOther>& other) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) >= other[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of less than operator to compare the array elements againt the
   * given value.
   * \param  val The value to compare to.
   * \tparam T   The type of the value.
   * \return true if all array elements are less than the given value.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator<(T val) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) >= val) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of greater than operator to compare one array to another.
   * \param  other     The other array to compare against this one.
   * \tparam ImplOther The implementation type of the other array.
   * \return true if all elements of this array are greater than the elements in
   *         the other array.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator>(const Array<ImplOther>& a) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) <= a[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Overload of greater than operator to compare the elements of the array to
   * given value.
   * \param  val The value to compare to.
   * \tparam T   The type of the value.
   * \return true if all elements in the array are greater than the given value.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator>(T val) const noexcept -> bool {
    for (size_t i = 0; i < elements; ++i) {
      if (impl()->operator[](i) <= val) {
        return false;
      }
    }
    return true;
  }

  /*==--- [operator {+,+=} overloads] --------------------------------------==*/

  /**
   * Overload of operator+= to add each element of the other array to this
   * array.
   * \note If the sizes of the arrays are different, this will cause a
   *       compile time error.
   *
   * \param  other     The array to add with.
   * \tparam ImplOther The implementation type of the other array.
   * \return A reference to the modified array.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator+=(const Array<ImplOther>& other) noexcept -> Impl& {
    assert_size_match<ImplOther>();
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) += other[i]; });
    return *impl();
  }

  /**
   * Overload of operator+= to add the value to each element of this
   * array.
   *
   * \note If the type T cannot be converted to the value type of the array
   *       then this will cause a compile time error.
   *
   * \param  val  The value to add to each element of the array.
   * \tparam T    The type of the value.
   * \return A reference to the modified array.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator+=(T val) noexcept -> Impl& {
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) += static_cast<Value>(val); });
    return *impl();
  }

  /**
   * Overload of operator+ to add each element of the other array to this array.
   *
   * \param  other     The array for the addition.
   * \tparam ImplOther The implementation type of the other array.
   * \return A new array with either this implementation type, or the
   *         implemenation type of the other array, or the fallback type.
   */
  template <typename ImplOther, typename R = array_impl_t<Impl, ImplOther>>
  ripple_all constexpr auto
  operator+(const Array<ImplOther>& other) const noexcept -> R {
    R result;
    unrolled_for_bounded<elements>(
      [&](auto i) { result[i] = impl()->operator[](i) + other[i]; });
    return result;
  }

  /**
   * Overload of operator+ to add the value val to each element of this
   * array and return a new array.
   *
   * \note If the type T cannot be converted to the value type of the
   *       array then this will cause a compile time error.
   *
   * \param  val  The value to add to each element of the array.
   * \tparam T    The type of the value.
   * \return A new array with either this implementation type, or the fallback
   *         type.
   */
  template <
    typename T,
    typename R                    = array_impl_t<Impl, Impl>,
    array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator+(T val) const noexcept -> R {
    R result;
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>([&](auto i) {
      result[i] = impl()->operator[](i) + static_cast<Value>(val);
    });
    return result;
  }

  /*==--- [operator {-,-=} overloads] --------------------------------------==*/

  /**
   * Overload of operator-= to subtract each element of the other array from
   * this array.
   *
   * \note If the sizes of the arrays are different, this will cause a compile
   *       time error.
   *
   * \param  other     The array to subtract with.
   * \tparam ImplOther The implementation type of the subtraction array.
   * \return A reference to the modified array.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator-=(const Array<ImplOther>& other) noexcept -> Impl& {
    assert_size_match<ImplOther>();
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) -= other[i]; });
    return *impl();
  }

  /**
   * Overload of operator-= to subtract the value from each element of this
   * array.
   *
   * \note If the type T cannot be converted to the value type of the array
   *       then this will cause a compile time error.
   *
   * \param  val  The value to subtract from each element of the array.
   * \tparam T    The type of the value.
   * \return A reference to the modified array.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator-=(T val) noexcept -> Impl& {
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) -= static_cast<Value>(val); });
    return *impl();
  }

  /**
   * Overload of operator- to subtract each element of the other array from each
   * element in this array, returning a new array.
   *
   *
   * \param  other     The array for the subtraction.
   * \tparam ImplOther The implementation type of the subtraction array.
   * \return A new array with either this implementation type, or the
   *         implemenation type of the other array, or the fallback type.
   */
  template <typename ImplOther, typename R = array_impl_t<Impl, ImplOther>>
  ripple_all constexpr auto
  operator-(const Array<ImplOther>& other) const noexcept -> R {
    R result;
    unrolled_for_bounded<elements>(
      [&](auto i) { result[i] = impl()->operator[](i) - other[i]; });
    return result;
  }

  /**
   * Overload of operator- to subtract the value from each element of this
   * array, returning a new array.
   *
   * \note If the type T cannot be converted to the value type of the array
   *       then this will cause a compile time error.
   *
   * \param  val  The value to subtract from each element of the array.
   * \tparam T    The type of the value.
   * \return A new array with either this implementation type, or the fallback
   *         type.
   */
  template <
    typename T,
    typename R                    = array_impl_t<Impl, Impl>,
    array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator-(T val) const noexcept -> R {
    R result;
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>([&](auto i) {
      result[i] = impl()->operator[](i) - static_cast<Value>(val);
    });
    return result;
  }

  /*==--- [operator {*,*=} overloads] --------------------------------------==*/

  /**
   * Overload of operator*= to multiply each element of the other array with
   * this one.
   *
   * \note If the sizes of the arrays are different, this will cause a compile
   *       time error.
   *
   * \param  other     The array to multiply with.
   * \tparam ImplOther The implementation type of the multiplication array.
   * \return A reference to the modified array.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator*=(const Array<ImplOther>& other) noexcept -> Impl& {
    assert_size_match<ImplOther>();
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) *= other[i]; });
    return *impl();
  }

  /**
   * Overload of operator*= to multiply the each element of the array with the
   * value.
   *
   * \note If the type T cannot be converted to the value type of the array then
   *       this will cause a compile time error.
   *
   * \param  val  The value to multiply with each element of the array.
   * \tparam T    The type of the value.
   * \return A reference to the modified array.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator*=(T val) noexcept -> Impl& {
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) *= static_cast<Value>(val); });
    return *impl();
  }

  /**
   * Overload of operator* to multiply each element of the array with the other
   * array, returning a new array.
   *
   *
   * \param  other     The array to multiply with.
   * \tparam ImplOther The implementation type of the multiplication array.
   * \return A new array with either this implementation type, or the
   *         implemenation type of the other array, or the fallback type.
   */
  template <typename ImplOther, typename R = array_impl_t<Impl, ImplOther>>
  ripple_all constexpr auto
  operator*(const Array<ImplOther>& other) const noexcept -> R {
    R result;
    unrolled_for_bounded<elements>(
      [&](auto i) { result[i] = impl()->operator[](i) * other[i]; });
    return result;
  }

  /**
   * Overload of operator* to multiply the value with each element of this
   * array, returning a new array.
   *
   *
   * \note If the type T cannot be converted to the value type of the array
   *       then this will cause a compile time error.
   *
   * \param  val  The value to multiply with each element of the array.
   * \tparam T    The type of the value.
   * \return A new array with either this implementation type, or the fallback
   *         type.
   */
  template <
    typename T,
    typename R                    = array_impl_t<Impl, Impl>,
    array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator*(T val) const noexcept -> R {
    R result;
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>([&](auto i) {
      result[i] = impl()->operator[](i) * static_cast<Value>(val);
    });
    return result;
  }

  /*==--- [operator {/,/=} overloads] --------------------------------------==*/

  /**
   * Overload of operator/= to divide each element of the array with each
   * element of the other array.
   *
   * \note If the sizes of the arrays are different, this will cause a compile
   *       time error.
   *
   * \param  other     The array to divide by.
   * \tparam ImplOther The implementation type of the other array.
   * \return A reference to the modified array.
   */
  template <typename ImplOther>
  ripple_all constexpr auto
  operator/=(const Array<ImplOther>& other) noexcept -> Impl& {
    assert_size_match<ImplOther>();
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) /= other[i]; });
    return *impl();
  }

  /**
   * Overload of operator/= to divide each element in the array by the value.
   *
   * \note If the type T cannot be converted to the value type of the array
   *       then this will cause a compile time error.
   *
   * \param  val  The value to divide each element of the array by.
   * \tparam T    The type of the value.
   * \return A reference to the modified array.
   */
  template <typename T, array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator/=(T val) noexcept -> Impl& {
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>(
      [&](auto i) { impl()->operator[](i) /= static_cast<Value>(val); });
    return *impl();
  }

  /**
   * Overload of operator/ to divide each element of the array with each
   * element of the other array, returning a new array.
   *
   * \note If the sizes of the arrays are different, this will cause a compile
   *       time error.
   *
   * \param  other     The array to divide by.
   * \tparam ImplOther The implementation type of the division array.
   * \return A new array with either this implementation type, or the
   *         implemenation type of the other array, or the fallback type.
   */
  template <typename ImplOther, typename R = array_impl_t<Impl, ImplOther>>
  ripple_all constexpr auto
  operator/(const Array<ImplOther>& other) const noexcept -> R {
    R result;
    unrolled_for_bounded<elements>(
      [&](auto i) { result[i] = impl()->operator[](i) / other[i]; });
    return result;
  }

  /**
   * Overload of operator/ to divide each element of this array by the value.
   *
   * \note If the type T cannot be converted to the value type of the array then
   *       this will cause a compile time error.
   *
   * \param  val  The value to multiply with each element of the array.
   * \tparam T    The type of the value.
   * \return A new array with either this implementation type, or the fallback
   *         type.
   */
  template <
    typename T,
    typename R                    = array_impl_t<Impl, Impl>,
    array_value_enable_t<T, Impl> = 0>
  ripple_all constexpr auto operator/(T val) const noexcept -> R {
    R result;
    using Value = typename array_traits_t<Impl>::Value;
    unrolled_for_bounded<elements>([&](auto i) {
      result[i] = impl()->operator[](i) / static_cast<Value>(val);
    });
    return result;
  }

 private:
  /**
   * Gets a pointer to the implementation of the interface.
   * \return A pointer to the implementation type.
   */
  ripple_all constexpr auto impl() noexcept -> Impl* {
    return static_cast<Impl*>(this);
  }

  /**
   * Gets a pointer to constant implementation of the interface.
   * \return A pointer to a const implementation type.
   */
  ripple_all constexpr auto impl() const noexcept -> const Impl* {
    return static_cast<const Impl*>(this);
  }

  /**
   * Performs a compile-time check that the size of the other array type
   * has the same number of elements that this array does.
   * \tparam ImplOther The implementation type of the array to check.
   */
  template <typename ImplOther>
  ripple_all constexpr auto assert_size_match() const noexcept -> void {
    constexpr size_t size_other = array_traits_t<ImplOther>::size;
    static_assert(
      size_other == elements, "Arrays have different number of elements");
  }
};

/*==--- [operator overloads] -----------------------------------------------==*/

/**
 * Overload of operator+ to add the scalar value to each element in the array,
 * returning a new array.
 *
 * \param  val   The value to add with each element of the array.
 * \param  a     The array to add with the value.
 * \tparam T     The type of the scalar.
 * \tparam Impl  The implementation type of the array.
 * \return A new array with either this implementation type, or the fallback
 *         type.
 */
template <
  typename T,
  typename Impl,
  typename R                    = array_impl_t<Impl, Impl>,
  array_value_enable_t<T, Impl> = 0>
ripple_all constexpr auto
operator+(T val, const Array<Impl>& a) noexcept -> R {
  using Value = typename array_traits_t<Impl>::Value;
  using Type  = std::decay_t<T>;

  static_assert(
    std::is_same_v<Type, Value> || std::is_convertible_v<Type, Value>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!");

  R result;
  unrolled_for_bounded<array_traits_t<Impl>::size>(
    [&](auto i) { result[i] = static_cast<Value>(val) + a[i]; });
  return result;
}

/**
 * Overload of operator- to subtract each element of the array from the scalar
 * value, returning a new array.
 *
 * \param  val   The value to subtract from.
 * \param  a     The array to subtract with.
 * \tparam T     The type of the scalar.
 * \tparam Impl  The implementation type of the array.
 * \return A new array with either this implementation type, or the fallback
 *         type.
 */
template <
  typename T,
  typename Impl,
  typename R                    = array_impl_t<Impl, Impl>,
  array_value_enable_t<T, Impl> = 0>
ripple_all constexpr auto
operator-(T val, const Array<Impl>& a) noexcept -> R {
  using Value = typename array_traits_t<Impl>::Value;
  using Type  = std::decay_t<T>;

  static_assert(
    std::is_same_v<Type, Value> || std::is_convertible_v<Type, Value>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!");

  R result;
  unrolled_for_bounded<array_traits_t<Impl>::size>(
    [&](auto i) { result[i] = static_cast<Value>(val) - a[i]; });
  return result;
}

/**
 * Overload of operator* to multiply each element of the array with the value,
 * returning a new array.
 *
 * \param  val   The value to multiply with the arry..
 * \param  a     The array to multiply with the scalar.
 * \tparam T     The type of the scalar.
 * \tparam Impl  The implementation type of the array.
 * \return A new array with either this implementation type, or the fallback
 *         type.
 */
template <
  typename T,
  typename Impl,
  typename R                    = array_impl_t<Impl, Impl>,
  array_value_enable_t<T, Impl> = 0>
ripple_all constexpr auto
operator*(T val, const Array<Impl>& a) noexcept -> R {
  using Value = typename array_traits_t<Impl>::Value;
  using Type  = std::decay_t<T>;

  static_assert(
    std::is_same_v<Type, Value> || std::is_convertible_v<Type, Value>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!");

  R result;
  unrolled_for_bounded<array_traits_t<Impl>::size>(
    [&](auto i) { result[i] = static_cast<Value>(val) * a[i]; });
  return result;
}

/**
 * Overload of operator/ to divide the scalar value by each element of the
 * array, returning a new array.
 *
 * \param  val   The value to divide.
 * \param  a     The array to divide by.
 * \tparam T     The type of the scalar.
 * \tparam Impl  The implementation type of the array.
 * \return A new array with either this implementation type, or the fallback
 *         type.
 */
template <
  typename T,
  typename Impl,
  typename R                    = array_impl_t<Impl, Impl>,
  array_value_enable_t<T, Impl> = 0>
ripple_all constexpr auto
operator/(T val, const Array<Impl>& a) noexcept -> R {
  using Value = typename array_traits_t<Impl>::Value;
  using Type  = std::decay_t<T>;

  static_assert(
    std::is_same_v<Type, Value> || std::is_convertible_v<Type, Value>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!");

  R result;
  unrolled_for_bounded<array_traits_t<Impl>::size>(
    [&](auto i) { result[i] = static_cast<Value>(val) / a[i]; });
  return result;
}

} // namespace ripple

#endif // RIPPLE_CONTAINER_ARRAY_HPP

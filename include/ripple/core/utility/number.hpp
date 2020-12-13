//==--- ripple/core/utility/number.hpp --------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  number.hpp
/// \brief This file defines a class which represents a compile time constant
///        number type, where a number can be represented as a time, and also
///        have a value at runtime.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_NUMBER_HPP
#define RIPPLE_UTILITY_NUMBER_HPP

#include "portability.hpp"

namespace ripple {

/**
 * The Num struct stores a number as a type, but also provides functionality to
 * convert the value of the number into a numeric type at runtime or compile
 * time. It is useful in a metaprogramming context to use as a type, but also
 * when the numeric value of the type is required. It works better with
 * variadic templates than using a raw numeric literal.
 * \tparam Value The value of the number.
 */
template <std::size_t Value>
struct Num {
  /** Returns the value of the number. */
  static constexpr auto value = size_t{Value};

  /** Conversion to size_t so that the number can be used as a size type. */
  ripple_host_device constexpr operator size_t() const noexcept {
    return Value;
  }
};

/**
 * The Int64 struct stores an integer as a type, but also provides
 * functionality to convert the value of the Int64 into a 64 bit integer type
 * at runtime or compile time. It is useful in a metaprogramming context to use
 * as a type, but also when the numeric value of the type is required. It works
 * better with variadic templates than using a raw numeric literal.
 * \tparam Value The value of the number.
 */
template <int64_t Value>
struct Int64 {
  /** Returns the value of the index. */
  static constexpr auto value = int64_t{Value};

  /** Conversion to int64_t so that the number can be used as a size type. */
  ripple_host_device constexpr operator int64_t() const {
    return Value;
  }
};

namespace detail {

/**
 * Struct to determine if T is a number.
 * \param T The type to determine if is a number.
 */
template <typename T>
struct IsNumber {
  /** Returns that T is not a dimension. */
  static constexpr bool value = false;
};

/**
 * Specialization for a number.
 * \param Value The value of the number.
 */
template <size_t Value>
struct IsNumber<Num<Value>> {
  /** Returns that T is a dimension. */
  static constexpr bool value = true;
};

/**
 * Specialization for a number.
 * \param Value The value of the number.
 */
template <int64_t Value>
struct IsNumber<Int64<Value>> {
  /** Returns that T is a dimension. */
  static constexpr bool value = true;
};

/**
 * Returns true if the type T is a number.
 * \tparam T The type to detrmine if is a number.
 */
template <typename T>
static constexpr bool is_number_v = detail::IsNumber<std::decay_t<T>>::value;

} // namespace detail

} // namespace ripple

#endif // RIPPLE_UTILITY_NUMBER_HPP

//==--- ripple/core/utility/dim.hpp ------------------------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dim.hpp
/// \brief This file defines a class which represents a dimension which can be
///        evaluated at compile time or as a size type at runtime.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_DIM_HPP
#define RIPPLE_UTILITY_DIM_HPP

#include "portability.hpp"

namespace ripple {

/**
 * Defines a class to represent a dimension, where the value is known at
 * compile time, but can also be evaluated at runtime as a size type.
 *
 * The class should be used through the aliases when a single dimension must be
 * specified to make code more readible:
 *
 * ~~~cpp
 * // Not clear, what is 0?
 * do_something(container, 0);
 *
 * // More clear, intention of x dimension application is known at call site.
 * do_something(container, dimx());
 * ~~~
 *
 * The other use case is when used with `unrolled_for`, where is can be used
 * more generically in a constexpr context:
 *
 * ~~~cpp
 * unrolled_for<num_dimensions>([] (auto dim_value) {
 *   // Compile time dimension can be used here:
 *   do_something(container, Dimension<dim_value>{});
 * });
 * ~~~
 *
 * or explicitly made `constexpr` (this can improve access performance):
 *
 * ~~~cpp
 * unrolled_for<num_dimensions>([] (auto dim_value) {
 *   constexpr auto dim = dim_value;
 *
 *   // Access the dim element of a container:
 *   auto v = container[dim];
 *
 *   // Use in a template type:
 *   auto block = block_with_cx_padding<dim>(...);
 * });
 * ~~~
 *
 * \tparam Value The value of the dimension.
 */
template <size_t Value>
struct Dimension {
  /** Returns the value of the dimension. */
  static constexpr size_t value = Value;

  /**
   * Overload of operator size_t to convert a dimension to a size_t.
   * \return The value of the dimension.
   */
  ripple_host_device constexpr operator size_t() const {
    return static_cast<size_t>(Value);
  }
};

namespace detail {

/**
 * Struct to determine if T is a Dimension type.
 * \param T The type to determine if is a dimension.
 */
template <typename T>
struct IsDimension {
  /** Returns that T is not a dimension. */
  static constexpr bool value = false;
};

/**
 * Specialization for a dimension type.
 * \param Value The value of the dimension.
 */
template <size_t Value>
struct IsDimension<Dimension<Value>> {
  /** Returns that T is a dimension. */
  static constexpr bool value = true;
};

} // namespace detail

/*==--- [aliases] ----------------------------------------------------------==*/

/** Aliases for the x spacial dimension type. */
using dimx_t = Dimension<0>;
/** Alias for the y spacial dimension type. */
using dimy_t = Dimension<1>;
/** Aloas for the z spacial dimension type. */
using dimz_t = Dimension<2>;

/*==--- [constants] --------------------------------------------------------==*/

/**
 * Defines a compile time type for the x spatial dimension.
 * \return A x dimension type.
 */
static constexpr inline dimx_t dimx() {
  return dimx_t{};
}

/**
 * Defines a compile time type for the y spatial dimension.
 * \return A y dimension type.
 */
static constexpr inline dimy_t dimy() {
  return dimy_t{};
}

/**
 * Defines a compile time type for the z spatial dimension.
 * \return A z dimension type.
 */
static constexpr inline dimz_t dimz() {
  return dimz_t{};
}

/**
 * Returns true if T is a Dimension, otherwise returns false.
 * \tparam T The type to determine if is a dimension.
 */
template <typename T>
static constexpr size_t is_dimension_v =
  detail::IsDimension<std::decay_t<T>>::value;

/**
 * Defines a valid type if the given template is a dimension.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using dim_enable_t = std::enable_if_t<is_dimension_v<T>, int>;

/**
 * Defines a valid type if the given template is not a dimension.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_dim_enable_t = std::enable_if_t<!is_dimension_v<T>, int>;

} // namespace ripple

#endif // RIPPLE_UTILITY_DIM_HPP

//==--- ripple/utility/dim.hpp ----------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
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


/// Defines a class to represent a dimension, where the value is known at
/// compile time, but can also be evaluated at runtime as a size type. The class
/// should be used through the aliases when a single dimension must be specified
/// to make code more readible:
/// 
/// ~~~cpp
/// // Not clear, what is 0?
/// do_something(container, 0);
/// 
/// // More clear, intention of x dimension application is known at call site.
/// do_something(container, dim_x);
/// ~~~
/// 
/// The other use case is when used with `unrolled_for`, where is can be used
/// more generically in a constexpr context:
/// 
/// ~~~cpp
/// unrolled_for<num_dimensions>([] (auto dim_value) {
///   // Compile time dimension can be used here:
///   do_something(container, Dimension<dim_value>{});
/// });
/// ~~~
///
/// or explicitly made `constexpr` (this can improve access performance):
///
/// ~~~cpp
/// unrolled_for<num_dimensions>([] (auto dim_value) {
///   constexpr auto dim = dim_value;
///
///   // Access the dim element of a container:
///   auto v = container[dim];
///
///   // Use in a template type:
///   auto block = block_with_cx_padding<dim>(...);
/// });
/// ~~~
/// 
/// \tparam Value The value of the dimension.
template <std::size_t Value>
struct Dimension {
  /// Returns the value of the dimension.
  static constexpr std::size_t value = Value;

  /// Overload of operator size_t to convert a dimension to a size_t.
  constexpr operator size_t() const {
    return static_cast<std::size_t>(Value);
  }
};

//==--- [aliases] ----------------------------------------------------------==//

/// Aliases for the x spacial dimension type.
using dimx_t = Dimension<0>;
/// Alias for the y spacial dimension type.
using dimy_t = Dimension<1>;
/// Aloas for the z spacial dimension type.
using dimz_t = Dimension<2>;

//==--- [constants] --------------------------------------------------------==//

/// Defines a compile time type for the x spacial dimension.
static constexpr dimx_t dim_x = dimx_t{};
/// Defines a compile time type for the x spacial dimension.
static constexpr dimy_t dim_y = dimy_t{};
/// Defines a compile time type for the x spacial dimension.
static constexpr dimz_t dim_z = dimz_t{};

} // namespace ripple

#endif // RIPPLE_UTILITY_DIM_HPP


//==--- ripple/fv/state/state_traits.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_traits.hpp
/// \brief This file defines an implementation for traits for states.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FV_STATE_STATE_TRAITS_HPP
#define RIPPLE_FV_STATE_STATE_TRAITS_HPP

#include <ripple/core/container/array_traits.hpp>
#include <ripple/core/storage/storage_layout.hpp>

namespace ripple {
namespace fv     {

//==--- [forwared declarations] --------------------------------------------==//
//
/// Forward declaration of the State static interface.
/// \tparam Impl The implementation type of the interface.
template <typename Impl> class State;

/// The FluidState type defines a class which stores state data representing
/// fluids.
/// \tparam T      The data type for the state data.
/// \tparam Dims   The number of dimensions for the fluid.
/// \tparam Layout The layout type for the data.
template <typename T, typename Dims, typename Layout = contiguous_owned_t>
class FluidState;

/// The StateTraits type defines traits for state types, which can be
/// specialized for implementations of different states.
/// \tparam T The type of the state to defines traits for.
template <typename T> struct StateTraits;

//==--- [specializations] --------------------------------------------------==//

/// Specialization of the state traits class for a fluid state.
/// \tparam T The type of the data for the state.
/// \tparam Dims The number of dimensions for the state.
/// \tparam Layout The type of the layout for the state.
template <typename T, typename Dims, typename Layout>
struct StateTraits<FluidState<T, Dims, Layout>> {
  /// Defines the number of dimensions for the state.
  static constexpr auto dimensions = Dims::value;
  /// Defines the total number of elements in the state.
  static constexpr auto elements   = dimensions + 2;

  /// Defines the type of the data for the state.
  using value_t             = std::decay_t<T>;
  /// Defines the type of the state with a contiguous layout.
  using contiguous_layout_t = FluidState<value_t, Dims, contiguous_owned_t>;
  /// Defiens the type of the layout for the state.
  using layout_t            = Layout;
  /// Defines type type of the flux vector.
  using flux_vec_t          = Vector<value_t, elements, contiguous_owned_t>;
};

/// Specialization of the state traits for the State interface.
/// \tparam Impl The implementation type of the state interface.
template <typename Impl>
struct StateTraits<State<Impl>> {
 private:
  /// Defines the traits for the implemenation type of the interface.
  using traits_t = StateTraits<Impl>;
 
 public:
  /// Defines the number of dimensions for the state.
  static constexpr auto dimensions = traits_t::dimensions;
  /// Defines the total number of elements in the state.
  static constexpr auto elements   = traits_t::elements;

  /// Defines the type of the data for the state.
  using value_t             = typename traits_t::value_t;
  /// Defines the type of the state with a contiguous layout.
  using contiguous_layout_t = typename traits_t::contiguous_layout_t;
  /// Defiens the type of the layout for the state.
  using layout_t            = typename traits_t::layout_t;
  /// Defines type type of the flux vector.
  using flux_vec_t          = typename traits_t::flux_vec_t;
};

//==--- [aliases] ----------------------------------------------------------==//

/// Alias for the traits of the type T, after decaying the type.
/// \tparam T The type to get the state traits for.
template <typename T>
using state_traits_t = StateTraits<std::decay_t<T>>;

} // namespace fv

/// Specialization of the ArrayTraits for a fluid state.
template <typename T, typename Dims, typename Layout>
struct ArrayTraits<fv::FluidState<T, Dims, Layout>> {
  /// The value type for the array.
  using value_t  = std::decay_t<T>;
  /// Defines the type of the layout for the array.
  using layout_t = Layout;
  /// Defines the type of the array.
  using array_t  = fv::FluidState<value_t, Dims, layout_t>;

  /// Returns the number of elements in the array. The fluid state stores
  /// density and (energy/pressure), and a velocity component for each
  /// dimension.
  static constexpr auto size = Dims::value + 2;
};

} // namespace ripple

#endif // RIPPLE_FV_STATE_STATE_TRAITS_HPP



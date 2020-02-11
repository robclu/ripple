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

/// The FluidState type defines a class which stores state data representing
/// fluids.
/// \tparam T      The data type for the state data.
/// \tparam Dims   The number of dimensions for the fluid.
/// \tparam Layout The layout type for the data.
template <typename T, typename Dims, typename Layout = contiguous_owned_t>
class FluidState;

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



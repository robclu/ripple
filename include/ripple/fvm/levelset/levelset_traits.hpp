//==--- ripple/fvm/levelset/levelset_traits.hpp ------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_traits.hpp
/// \brief This file defines an implementation for traits for levelsets.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FVM_LEVELSET_LEVELSET_TRAITS_HPP 
#define RIPPLE_FVM_LEVELSET_LEVELSET_TRAITS_HPP 

#include <ripple/core/container/array_traits.hpp>
#include <ripple/core/storage/storage_layout.hpp>

namespace ripple {
namespace fv     {

//==--- [forwared declarations] --------------------------------------------==//

/// The Levelset type defines a class which stores scalar data in an N
/// dimensional space to represent the distance from an interface.
///
/// \tparam T      The data type for the levelset data.
/// \tparam Dims   The number of dimensions for the levelset.
/// \tparam Layout The layout type for the data.
template <typename T, typename Dims, typename Layout = contiguous_owned_t>
class Levelset; 

/// The LevelsteTraits type defines traits for state types, which can be
/// specialized for implementations of different levelsets.
/// \tparam T The type of the levelset to defines traits for.
template <typename T> struct LevelsetTraits;

//==--- [specializations] --------------------------------------------------==//

/// Specialization of the levelset traits class for a fluid state.
/// \tparam T       The type of the data for the state.
/// \tparam Dims    The number of dimensions for the state.
/// \tparam Layout  The type of the layout for the state.
template <typename T, typename Dims, typename Layout>
struct LevelsetTraits<Levelset<T, Dims, Layout>> {
  /// Defines the number of dimensions for the state.
  static constexpr auto dimensions = Dims::value;
  /// Defines that the levelset is a scalar field with only 1 element.
  static constexpr auto elements   = 1;

  /// Defines the type of the data for the state.
  using value_t              = std::decay_t<T>;
  /// Defines the type of the levelset with a contiguous owned layout.
  using contiguous_owned_t   = Levelset<value_t, Dims, contiguous_owned_t>;
  /// Defines the type of the levelset with a contiguous view layout.
  using contiguous_view_t    = Levelset<value_t, Dims, contiguous_view_t>;
  /// Defines the type of the levelset with a contiguous strided layout.
  using contiguous_strided_t = Levelset<value_t, Dims, strided_view_t>;
  /// Defiens the type of the layout for the state.
  using layout_t             = Layout;
};

//==--- [aliases] ----------------------------------------------------------==//

/// Alias for the levelset traits of the type T, after decaying the type.
/// \tparam T The type to get the levelset traits for.
template <typename T>
using levelset_traits_t = LevelsetTraits<std::decay_t<T>>;

} // namespace fv

/// Specialization of the ArrayTraits for a levelset.
/// \tparam T       The type of the data for the state.
/// \tparam Dims    The number of dimensions for the state.
/// \tparam Layout  The type of the layout for the state./
template <typename T, typename Dims, typename Layout>
struct ArrayTraits<fv::Levelset<T, Dims, Layout>> {
  /// The value type for the array.
  using value_t  = std::decay_t<T>;
  /// Defines the type of the layout for the array.
  using layout_t = Layout;
  /// Defines the type of the array.
  using array_t  = fv::Levelset<value_t, Dims, layout_t>;

  /// Returns the number of elements in the array. The levelset is a scalar
  /// field, so stores just a single value.
  static constexpr auto size = 1;
};

} // namespace ripple

#endif // RIPPLE_FV_STATE_STATE_TRAITS_HPP



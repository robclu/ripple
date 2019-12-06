//==--- ripple/multidim/space_traits.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  space_traits.hpp
/// \brief This file defines traits for dimensional spaces.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MULTIDIM_SPACE_TRAITS_HPP
#define RIPPLE_MULTIDIM_SPACE_TRAITS_HPP

namespace ripple {

//==--- [forward declations] -----------------------------------------------==//

/// The MultidimSpace defines an interface for classes when define a
/// multidimensional space.
/// \tparam Impl The implementation of the interface.
template <typename Impl> struct MultidimSpace;

/// The DynamicMultidimSpace struct defines spatial information over multiple
/// dimensions, specifically the sizes of the dimensions and the steps required
/// to get from one element to another in the dimensions of the space. The
/// dynamic nature of the space means that it can be modified. It implements the
/// MultidimSpace interface.
/// \tparam Dimensions The number of dimensions.
template <std::size_t Dimensions> struct DynamicMultidimSpace;

/// The StaticMultidimSpace struct defines spatial information over multiple
/// dimensions, specifically the sizes of the dimensions and the steps required
/// to get from one element to another in the dimensions of the space. The
/// static nature of the space means that it can't be modified, and the size and
/// steps for the space are all known at compile time, which makes using this
/// space more efficient, all be it with less flexibility.
/// \tparam Sizes The sizes of the dimensions of the space.
template <std::size_t... Sizes> struct StaticMultidimSpace;

//==--- [traits declations] ------------------------------------------------==//

/// Defines a class for traits for multidimensional spaces.
/// \tparam Space The space to get the traits for.
template <typename Space> struct SpaceTraits {
  //==--- [traits] ---------------------------------------------------------==//
  
  /// Defines that the space has a single dimension.
  static constexpr size_t dimensions = 1;
};

/// Specialization of the space traits for static multidimensional space.
/// \tparam Sizes The sizes of the dimensions of the space.
template <std::size_t... Sizes>
struct SpaceTraits<StaticMultidimSpace<Sizes...>> {
  //==--- [traits] ---------------------------------------------------------==//
  
  /// Defines the number of dimensions for the iterator.
  static constexpr size_t dimensions = sizeof...(Sizes);
};

/// Specialization of the space traits for a dynamic multidimensional space.
/// \tparam Dimensions The number of dimensions for the space.
template <std::size_t Dimensions>
struct SpaceTraits<DynamicMultidimSpace<Dimensions>> {
  //==--- [traits] ---------------------------------------------------------==//
  
  /// Defines the number of dimensions for the iterator.
  static constexpr size_t dimensions = Dimensions;
};

/// Specialization of the space traits for any type implementing the multidim
/// interface.
/// \tparam SpaceImpl The implementation type.
template <typename SpaceImpl>
struct SpaceTraits<MultidimSpace<SpaceImpl>> {
  private:
   /// Defines the type of the traits for the implementation.
   using impl_traits_t = SpaceTraits<SpaceImpl>;

  public:
  //==--- [traits] ---------------------------------------------------------==//
  
  /// Defines the number of dimensions for the iterator.
  static constexpr size_t dimensions = impl_traits_t::dimensions;
};

//==--- [aliases] ----------------------------------------------------------==//

/// Defines an alias for space traits which decays the type T before determining
/// the traits.
/// \tparam T The type to get the space traits for.
template <typename T>
using space_traits_t = SpaceTraits<std::decay_t<T>>;

} // namespace ripple

#endif // RIPPLE_MULTIDIM_SPACE_TRAITS_HPP

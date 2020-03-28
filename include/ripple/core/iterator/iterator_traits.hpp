//==--- ripple/core/iterator/iterator_traits.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  iterator_traits.hpp
/// \brief This file defines traits for iterators.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ITERATOR_ITERATOR_TRAITS_HPP
#define RIPPLE_ITERATOR_ITERATOR_TRAITS_HPP

#include <ripple/core/multidim/space_traits.hpp>
#include <ripple/core/storage/storage_traits.hpp>

namespace ripple {

//==--- [forward declations] -----------------------------------------------==//

/// The BlockIterator class defines a iterator over a block, for a given space
/// which defines the region of the block. The iterator only iterates over the
/// internal space of the block, not over the padding.
///
/// The type T for the iterator can be e, or type which
/// implements the StridableLayout interface. Regardless, the use is the same,
/// and the iterator operator as if it was a pointer to T.
///
/// \tparam T     The data type which the iterator will access.
/// \tparam Space The type which defines the iteration space.
template <typename T, typename Space> class BlockIterator;

//==--- [traits declations] ------------------------------------------------==//

/// Defines a class for traits for iterators.
/// \tparam Iterator The iterator to get traits for.
template <typename Iterator> struct IteratorTraits {
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the value type of the iterator.
  using value_t = void*;

  //==--- [traits] ---------------------------------------------------------==//
  
  /// Defines that the iterator has a single dimension.
  static constexpr size_t dimensions = 1;
  /// Returns that the traits are not for a valid iterator.
  static constexpr bool  is_iterator = false;
};

/// Specialization of the iterator traits for a block iterator.
/// \tparam T     The type of the block iterator.
/// \tparam Space The space for the iterator.
template <typename T, typename Space>
struct IteratorTraits<BlockIterator<T, Space>> {
 private:
   /// Defines the layout traits for the iterator.
  using layout_traits_t = layout_traits_t<T>;

 public:
  /// Defines the value type of the iterator.
  using value_t = typename layout_traits_t::value_t;
  /// Defines the copy type for the iterator, which is the type which ensures
  /// that the iterator data is copied.
  using copy_t  = typename layout_traits_t::iter_copy_t;
  /// Defines the reference type for the iterator.
  using ref_t   = typename layout_traits_t::iter_ref_t;

  //==--- [traits] ---------------------------------------------------------==//
  
  /// Defines the number of dimensions for the iterator.
  static constexpr size_t dimensions  = space_traits_t<Space>::dimensions;
  /// Returns that the traits are for a valid iterator.
  static constexpr bool   is_iterator = true;
};

//==--- [aliases] ----------------------------------------------------------==//

/// Defines the iterator traits for the type T after decaying the type T.
/// \tparam T The type to get the iterator traits for.
template <typename T>
using iterator_traits_t = IteratorTraits<std::decay_t<T>>;

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T is an iterator.
/// \tparam T The type to determine if is an iterator.
template <typename T>
static constexpr auto is_iterator_v = iterator_traits_t<T>::is_iterator;

//==--- [enables] ----------------------------------------------------------==//

/// Defines a valid type if the type T is an iterator.
/// \tparam T The type to base the enable on.
template <typename T>
using iterator_enable_t = std::enable_if_t<is_iterator_v<T>, int>;

/// Defines a valid type if the type T is not an iterator.
/// \tparam T The type to base the enable on.
template <typename T>
using non_iterator_enable_t = std::enable_if_t<!is_iterator_v<T>, int>;

/// Defines a valid type if the type T is an iterator, and the iterator has one
/// dimension.
/// \tparam T The type to base the enable on.
template <typename T>
using it_1d_enable_t = std::enable_if_t<
  is_iterator_v<T> && iterator_traits_t<T>::dimensions == 1, int
>;

/// Defines a valid type if the type T is an iterator, and the iterator has two
/// dimensions.
/// \tparam T The type to base the enable on.
template <typename T>
using it_2d_enable_t = std::enable_if_t<
  is_iterator_v<T> && iterator_traits_t<T>::dimensions == 2, int
>;

/// Defines a valid type if the type T is an iterator, and the iterator has
/// three dimensions.
/// \tparam T The type to base the enable on.
template <typename T>
using it_3d_enable_t = std::enable_if_t<
  is_iterator_v<T> && iterator_traits_t<T>::dimensions == 3, int
>;

} // namespace ripple

#endif // RIPPLE_ITERATOR_ITERATOR_TRAITS_HPP

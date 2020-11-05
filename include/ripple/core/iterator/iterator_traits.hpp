//==--- ripple/core/iterator/iterator_traits.hpp ----------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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

#include <ripple/core/container/array_traits.hpp>
#include <ripple/core/multidim/space_traits.hpp>
#include <ripple/core/storage/storage_traits.hpp>

namespace ripple {

/*==--- [forward declations] -----------------------------------------------==*/

/**
 * The BlockIterator class defines a iterator over a block, for a given space
 * which defines the region of the block. The iterator only iterates over the
 * internal space of the block, not over the padding, and does not have any
 * knowledge of the where the block is in the global context, or where the
 * iterator is in the block. It is ideal for cases where such information is
 * not required, and operations are relative to the iterator (i.e stencil-like
 * operations, operations which required neighbour data, and work on shared
 * memory data).
 *
 * The type T for the iterator can be either a normal type, or type which
 * implements the StridableLayout interface. Regardless, the use is the same,
 * and the iterator operator as if it was a pointer to T.
 *
 * \tparam T      The data type which the iterator will access.
 * \tparam Space  The type which defines the iteration space.
 */
template <typename T, typename Space>
class BlockIterator;

/**
 * The IndexedIterator class is an extended BlockIterator, with additional
 * information which provides it with context for where the block is relative
 * to the global space to which it belongs. It also allows the thread indidces
 * to be determined.
 *
 * For this iterator, the global size and index data allow the iterator to be
 * used to implement consistent interfaces for iteration across a global space
 * which is split across multiple devices, which is the primary purpose of the
 * class.
 *
 * The type T for the iterator can be either a normal type, or a type which
 * implements the StridableLayout interface, which allows the data to be stored
 * and iterated as SoA instead of AoS. Regardless, the use is the same,
 * and the iterator semantics are as if it was a poitner to T.
 *
 * \tparam T      The data type which the iterator will access.
 * \tparam Space  The type which defines the iteration space.
 */
template <typename T, typename Space>
class IndexedIterator;

/*==--- [traits declations] ------------------------------------------------==*/

/**
 * Defines a class for traits for iterators. This is the default case for types
 * which are not iterators.
 * \tparam Iterator The iterator to get traits for.
 */
template <typename Iterator>
struct IteratorTraits {
  /** Defines the value type of the iterator. */
  using Value = void*;

  /** Defines that the iterator has a single dimension. */
  static constexpr size_t dimensions = 1;
  /** Returns that the traits are not for a valid iterator. */
  static constexpr bool is_iterator = false;
};

/**
 * Specialization of the iterator traits for a block iterator.
 * \tparam T     The type of the block iterator.
 * \tparam Space The space for the iterator.
 */
template <typename T, typename Space>
struct IteratorTraits<BlockIterator<T, Space>> {
 private:
  /** Defines the layout traits for the iterator. */
  using LayoutTraits = layout_traits_t<T>;

 public:
  // clang-format off

  /** Defines the number of dimensions for the iterator. */
  static constexpr size_t dimensions = space_traits_t<Space>::dimensions;
  /** Returns that the traits are for a valid iterator. */
  static constexpr bool is_iterator  = true;
  /** Returns that the iterator does not have index information. */
  static constexpr bool has_indices  = false;

  /** Defines the value type of the iterator. */
  using Value = typename LayoutTraits::Value;
  /** Defines the reference type for the iterator. */
  using Ref   = typename LayoutTraits::IterRef;
  /** 
   * Defines the copy type for the iterator, which is a type to ensure
   * that the iterator data is copied. 
  */
  using CopyType = typename LayoutTraits::IterCopy;
  /** Defines the type of a vector of value type with matching dimensions. */
  using Vec      = Vec<CopyType, dimensions, ContiguousOwned>;
  // clang-format on
};

/**
 * Specialization of the iterator traits for an indexed iterator.
 * \tparam T     The type of the block iterator.
 * \tparam Space The space for the iterator.
 */
template <typename T, typename Space>
struct IteratorTraits<IndexedIterator<T, Space>> {
 private:
  /** Defines the traits for the block iterator. */
  using BlockIterTraits = IteratorTraits<BlockIterator<T, Space>>;

 public:
  // clang-format off
  /** Defines the number of dimensions for the iterator. */
  static constexpr size_t dimensions = BlockIterTraits::dimensions;
  /** Returns that the traits are for a valid iterator. */
  static constexpr bool is_iterator  = true;
  /** Returns that the iterator does have index information. */
  static constexpr bool has_indices  = true;

  /** Defines the value type of the iterator. */
  using Value    = typename BlockIterTraits::Value;
  /** Defines the reference type for the iterator. */
  using Ref      = typename BlockIterTraits::Ref;
  /** Defines the type of a vector of value type with matching dimensions. */
  using Vec      = typename BlockIterTraits::Vec;
  /** Defines the copy type for the iterator. */
  using CopyType = typename BlockIterTraits::CopyType;
  // clang-format on
};

/*==--- [aliases] ----------------------------------------------------------==*/

/**
 * Defines the iterator traits for the type T after decaying the type T.
 * \tparam T The type to get the iterator traits for.
 */
template <typename T>
using iterator_traits_t = IteratorTraits<std::decay_t<T>>;

/*==--- [traits] -----------------------------------------------------------==*/

/**
 * Returns true if the type T is an iterator.
 * \tparam T The type to determine if is an iterator.
 */
template <typename T>
static constexpr auto is_iterator_v = iterator_traits_t<T>::is_iterator;

/*==--- [enables] ----------------------------------------------------------==*/

/**
 * Defines a valid type if the type T is an iterator.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using iterator_enable_t = std::enable_if_t<is_iterator_v<T>, int>;

/**
 * Defines a valid type if the type T is not an iterator.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_iterator_enable_t = std::enable_if_t<!is_iterator_v<T>, int>;

/**
 * Defines a valid type if the type T is an iterator, and the iterator has one
 * dimension.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using it_1d_enable_t = std::
  enable_if_t<is_iterator_v<T> && iterator_traits_t<T>::dimensions == 1, int>;

/**
 * Defines a valid type if the type T is an iterator, and the iterator has two
 * dimensions.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using it_2d_enable_t = std::
  enable_if_t<is_iterator_v<T> && iterator_traits_t<T>::dimensions == 2, int>;

/**
 * Defines a valid type if the type T is an iterator, and the iterator has
 * three dimensions.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using it_3d_enable_t = std::
  enable_if_t<is_iterator_v<T> && iterator_traits_t<T>::dimensions == 3, int>;

} // namespace ripple

#endif // RIPPLE_ITERATOR_ITERATOR_TRAITS_HPP

//==--- ripple/core/container/indexed_iterator.hpp --------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  indexed_iterator.hpp
/// \brief This file implements an iterator over a block, which has knowledge
///        of the block's relative location in the global space to which it
///        is a subregion of.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_INDEXED_ITERATOR_HPP
#define RIPPLE_CONTAINER_INDEXED_ITERATOR_HPP

#include "block_iterator.hpp"
#include <ripple/core/container/vec.hpp>
#include <ripple/core/execution/thread_index.hpp>

namespace ripple {

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
 * \todo The storage of the additional index and size data for the iterator
 *       increases the register usage in device code, which needs to be
 *       improved.
 *
 * \tparam T      The data type which the iterator will access.
 * \tparam Space  The type which defines the iteration space.
 */
template <typename T, typename Space>
class IndexedIterator : public BlockIterator<T, Space> {
  /** Defines the underlying type for the iterator. */
  using Value = typename layout_traits_t<T>::Value;

 public:
  // clang-format off
  /** Defines the type of the base class. */
  using BlockIter = BlockIterator<T, Space>;
  /** Defines the index type used for the iterator. */
  using Index     = uint32_t;
  /** Defines the type of the contianer used to store the indices. */
  using Indices   = Vec<Index, BlockIter::dims>;

  /** The number of dimensions for the iterator. */
  static constexpr size_t dims = BlockIter::dims;

  /** Defines the type for the coordinates of the iterator. */
  using Coord  = Vec<Index, dims>;
  /** Defines the type for the normalized coordinates of the iterator. */
  using Coordf = Vec<float, dims>;
  // clang-format on

  //==--- [consturction] ---------------------------------------------------==//

  /** Inherit all the base constructors. */
  using BlockIter::BlockIter;

  /**
   * Constuctor to copy a BlockIterator \p block_iter into this iterator.
   * \param block_iter The block iterator to use to set this iterator.
   */
  ripple_host_device IndexedIterator(BlockIter block_iter) noexcept
  : BlockIter{block_iter} {}

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Offsets the iterator by \p amount positions in the block in the \p dim
   * dimension.
   * \param  dim    The dimension to offset in
   * \param  amount The amount to offset by.
   * \tparam Dim    The type of the dimension specifier.
   * \return A new iterator to the offset location.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  offset(Dim&& dim, int amount = 1) const noexcept -> IndexedIterator {
    auto res =
      IndexedIterator{BlockIter::offset(static_cast<Dim&&>(dim), amount)};
    res.block_start_indices_ = block_start_indices_;
    res.global_sizes_        = global_sizes_;
    return res;
  }

  /**
   * Gets the thread index of the start of the block which this iterator can
   * iterate over.
   * \param  dim The dimension to get the block start index for.
   * \tparam Dim The type of the dimension specifier.
   * \return The first thread index in the dimension which this iterator can
   *         iterate from.
   */
  template <typename Dim>
  ripple_host_device auto block_start_index(Dim&& dim) const noexcept -> Index {
    return block_start_indices_.at(static_cast<Dim&&>(dim));
  }

  /**
   * Sets the index of the first thread which this iterator can iterate from in
   * the iteration space for the given dimension.
   * \param  dim   The dimension to set the block index for.
   * \param  index The index to set the start of the block to.
   * \tparam Dim   The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_host_device auto
  set_block_start_index(Dim&& dim, Index index) noexcept -> void {
    block_start_indices_.at(static_cast<Dim&&>(dim)) = index;
  }

  /**
   * Gets the size of the global space to which the block which can be iterated
   * over can belongs for a given dimension (the total number of threads).
   * \param  dim The dimension to get the global size for.
   * \tparam Dim The type of the dimension specifier.
   * \return The number of elements in the dimension in the global space to
   *         which this iterator belongs.
   */
  template <typename Dim>
  ripple_host_device auto global_size(Dim&& dim) const noexcept -> Index {
    return global_sizes_.at(static_cast<Dim&&>(dim));
  }

  /**
   * Sets the size of the global space to which the block belongs for
   * dimension \p dim to \p size.
   * \param  dim  The dimension to set the size for.
   * \param  size The size for the dimension.
   * \tparam Dim  The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_host_device auto
  set_global_size(Dim&& dim, Index index) noexcept -> void {
    global_sizes_.at(static_cast<Dim&&>(dim)) = index;
  }

  /**
   * Gets the global index of the data to which this iterator points, in
   * the global space for which this iterator can iterate a sub region.
   * \param  dim The dimension to get the global index for.
   * \tparam Dim The type of the dimension specifier.
   * \return The index of this iterator in the global space.
   */
  template <typename Dim>
  ripple_host_device auto global_idx(Dim&& dim) const noexcept -> size_t {
    return ::ripple::global_idx(static_cast<Dim&&>(dim)) +
           static_cast<size_t>(block_start_indices_[dim]);
  }

  /**
   * Gets the global normalized index of the pointed to element for the
   * iterator in the \p dim dimension.
   * \param  dim The dimension to get the global normalzies index for.
   * \tparam Dim The type of the dimension specifier.
   * \return The global normalized index of the iterator in the global iteration
   *         space.
   *
   */
  template <typename Dim>
  ripple_host_device auto normalized_idx(Dim&& dim) const noexcept -> double {
    return static_cast<double>(global_idx(static_cast<Dim&&>(dim))) /
           global_sizes_[dim];
  }

  /**
   * Determines  if the iterator is valid for the \p dim dimension. It is valid
   * if its index is less than the number of elements in the dimension.
   * \param  dim The dimension to check validity in.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the iterator is valid for the dimension.
   */
  template <typename Dim>
  ripple_host_device auto is_valid(Dim&& dim) const noexcept -> bool {
    //    const auto global_idx_in_block = ::ripple::global_idx(dim);
    //    return global_idx_in_block < BlockIter::size(dim) &&
    //           (global_idx_in_block + block_start_indices_[dim]) <
    //             global_sizes_[dim];
    return block_start_indices_[dim] +
             ::ripple::global_idx(static_cast<Dim&>(dim)) <
           global_sizes_[dim];
  }

  /*==--- [coordinates] ----------------------------------------------------==*/

  /**
   * Gets the co-ordinates of the iterator in the global space.
   * \return The co-ordinates of the iterator in the global space.
   */
  ripple_host_device auto coord() noexcept -> Coord {
    return make_coord(std::make_index_sequence<dims>());
  }

  /**
   * Gets the co-ordinates of the iterator in the global space normalized to the
   * 0 -> 1 range.
   * \return The co-ordinates of the iterator in the global space.
   */
  ripple_host_device auto norm_coord() noexcept -> Coordf {
    return make_norm_coord(std::make_index_sequence<dims>());
  }

  /**
   * Determines if this is the first iterator in the global space, that is, it's
   * the first valid iterator in each dimensions of the space.
   * \return true if this is the first iterator in the global space.
   */
  ripple_host_device auto first_in_global_space() const noexcept -> bool {
    for (size_t dim = 0; dim < dims; ++dim) {
      if (global_idx(dim) != 0) {
        return false;
      }
    }
    return true;
  }

  /**
   * Determines if this is the last iterator in the global space, that is, it's
   * the last valid iterator in each dimensions of the space.
   * \return true if this is the last iterator in the global space.
   */
  ripple_host_device auto last_in_global_space() const noexcept -> bool {
    for (size_t dim = 0; dim < dims; ++dim) {
      if (global_idx(dim) != (global_sizes_[dim] - 1)) {
        return false;
      }
    }
    return true;
  }

 private:
  Indices block_start_indices_{0}; //!<  Indices of the iterable block;
  Indices global_sizes_{0};        //!< Global sizes of the space.

  /**
   * Makes the global coordinates for the iterator.
   * \tparam I The indices of each dimension to get the coordinate for.
   * \return The coordinates of the iterator.
   */
  template <size_t... I>
  ripple_host_device auto
  make_coord(std::index_sequence<I...>) const noexcept -> Coord {
    return Coord{global_idx(Dimension<I>())...};
  }

  /**
   * Makes the global normalized coordinates for the iterator.
   * \tparam I The indices of each dimension to get the coordinate for.
   * \return The coordinates of the iterator.
   */
  template <size_t... I>
  ripple_host_device auto
  make_norm_coord(std::index_sequence<I...>) const noexcept -> Coordf {
    return Coordf{normalized_idx(Dimension<I>())...};
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_INDEXED_ITERATOR_HPP

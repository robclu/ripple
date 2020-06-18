//==--- ripple/core/container/indexed_iterator.hpp --------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas
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

namespace ripple {

/// The IndexedIterator class is an extended BlockIterator, with additional
/// information which provides it with context for where the block is relative
/// to the global space to which it belongs. It also allows the thread indidces
/// to be determined.
///
/// The type T for the iterator can be either a normal type, or type which
/// implements the StridableLayout interface. Regardless, the use is the same,
/// and the iterator operats as if it was a pointer to T.
///
/// \tparam T      The data type which the iterator will access.
/// \tparam Space  The type which defines the iteration space.
template <typename T, typename Space>
class IndexedIterator : public BlockIterator<T, Space> {
 public:
  // clang-format off
  /// Defines the type of the base class.
  using block_iter_t = BlockIterator<T, Space>;
  /// Defines the index type used for the iterator.
  using index_t      = uint32_t;
  /// Defines the type of the contianer used to store the indices.
  using indices_t    = Vec<index_t, block_iter_t::dims>;
  // clang-format on

  /// The number of dimensions for the iterator.
  static constexpr size_t dims = block_iter_t::dims;

  //==--- [consturction] ---------------------------------------------------==//

  /// Inherit all the base constructors.
  using block_iter_t::block_iter_t;

  /// Constuctor to copy a BlockIterator \p block_iter into this iterator.
  /// \param block_iter The block iterator to use to set this iterator.
  ripple_host_device IndexedIterator(block_iter_t block_iter)
  : block_iter_t{block_iter} {}

  //==--- [interface] ------------------------------------------------------==//

  /// Returns the index of the start of the block being iterated over in
  /// dimension \p dim.
  /// \param  dim The dimension to get the block start index for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto
  block_start_index(Dim&& dim) const noexcept -> index_t {
    return _block_indices.at(std::forward<Dim>(dim));
  }

  /// Sets the index of the first thread in the block being iterated over in
  /// dimension \p dim to \p index.
  /// \param  dim   The dimension to set the block index for.
  /// \param  index The index to set the start of the block to.
  /// \tparam Dim   The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto
  set_block_start_index(Dim&& dim, index_t index) noexcept -> void {
    _block_indices.at(std::forward<Dim>(dim)) = index;
  }

  /// Returns size of the global space to which the block belongs for dimension
  /// \p dim.
  /// \param  dim The dimension to get the global size for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto global_size(Dim&& dim) const noexcept -> index_t {
    return _global_sizes.at(std::forward<Dim>(dim));
  }

  /// Sets the size of the global space to which the block belongs for
  /// dimension \p dim to \p size.
  /// \param  dim  The dimension to set the size for.
  /// \param  size The size for the dimension.
  /// \tparam Dim  The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto
  set_global_size(Dim&& dim, index_t index) noexcept -> void {
    _global_sizes.at(std::forward<Dim>(dim)) = index;
  }

  /// Returns the global index of the pointed to element for the iterator in the
  /// \p dim dimension.
  /// \param  dim The dimension to get the global index for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto global_idx(Dim&& dim) const noexcept -> size_t {
    return ::ripple::global_idx(std::forward<Dim>(dim)) + _block_indices[dim];
  }

  /// Returns true if the iterator is valid for the \p dim dimension -- that its
  /// index is less than the size of the dimension.
  /// \param  dim The dimension to check validity in.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto is_valid(Dim&& dim) const noexcept -> bool {
    const auto global_idx_in_block = ::ripple::global_idx(dim);
    return global_idx_in_block < block_iter_t::size(dim) &&
           global_idx_in_block + _block_indices[dim] < _global_sizes[dim];
  }

 private:
  indices_t _block_indices{0}; //!<  Indices of the iterable block;
  indices_t _global_sizes{0};  //!< Global sizes of the space.
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_INDEXED_ITERATOR_HPP

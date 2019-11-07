//==--- ripple/container/host_block.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  host_block.hpp
/// \brief This file imlements a host side block class which holds some data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_HOST_BLOCK_HPP
#define RIPPLE_CONTAINER_HOST_BLOCK_HPP

#include "block_traits.hpp"
#include <ripple/iterator/block_iterator.hpp>
#include <ripple/multidim/dynamic_multidim_space.hpp>
#include <ripple/storage/storage_traits.hpp>
#include <cstring>

namespace ripple {

/// Implementation of a host block class which stores multidimensional data on
/// the host. This will store the data in a strided format if the type T
/// implements the StriableLayout interface and the descriptor for the storage
/// for the type has a StorageLayout::strided_view type, otherwise this will
/// store the data in a contiguous format.
///
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions>
class HostBlock {
  //==--- [traits] ---------------------------------------------------------==//

  /// Defines the type of this tensor.
  using self_t          = HostBlock<T, Dimensions>;
  /// Defines the type of the pointer to the data.
  using ptr_t           = void*;
  /// Defines the the of the dimension information for the tensor.
  using space_t         = DynamicMultidimSpace<Dimensions>;
  /// Defines the allocation traits for the type.
  using layout_traits_t = layout_traits_t<T>;
  /// Defines the type of allocator for the tensor.
  using allocator_t     = typename layout_traits_t::allocator_t;
  /// Defines the type for a reference to an element.
  using value_t         = typename layout_traits_t::value_t;

  /// Defines the type of the iterator for the tensor.
  using iter_t          = BlockIterator<value_t, space_t>;
  /// Defines the type of a contant iterator.
  using const_iter_t    = const iter_t;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor which just initializes the storage. This constructor
  /// is available so that an unsized tensor can be instanciated and then
  /// resized at runtime.
  HostBlock() = default;

  /// Destructor to clean up the tensor resources.
  ~HostBlock() {
    cleanup();
  }

  /// Initializes the size of each of the dimensions of the tensor. This is only
  /// enabled when the number of arguments matches the dimensionality of the
  /// tensor, and the sizes are numeric types.
  ///
  /// \param size_0 The size of the zero dimension for the tensor.
  /// \param sizes  The sizes of the other dimensions of the tensor, if there
  ///               are additional dimensions.
  /// \tparam Size  The type of the zero dimension size.
  /// \tparam Sizes The types of other dimension sizes.
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  HostBlock(Sizes&&... sizes)
  : _space{std::forward<Sizes>(sizes)...} {
    allocate();
  }

  /// Constructor to create the block from another block.
  /// \param[ other The other block to create this block from.
  ripple_host_device HostBlock(const self_t& other)
  : _space{other._space} {
    allocate();
    const auto bytes = allocator_t::allocation_size(_space.size());
    std::memcpy(_data, other._data, bytes);
  }

  /// Constructor to move the \p other block into this one. The \p other will no
  /// longer have valid data, and it's size is meaningless.
  /// \param other The other block to move to this one.
  ripple_host_device HostBlock(self_t&& other)
  : _space{other._space} {
    _data       = other._data;
    other._data = nullptr;
  }

  //==--- [operator overloading] -------------------------------------------==//

  /// Overload of assignment operator to copy a block.
  /// \param[in] other The other block to create this block from.
  ripple_host_device auto operator=(const self_t& other) -> self_t& {
    _space = other._space;
    allocate();
    const auto bytes = allocator_t::allocation_size(_space.size());
    std::memcpy(_data, other._data, bytes);
    return *this;
  }

  /// Overload of assignment operator to copy a block.
  /// \param[in] other The other block to create this block from.
  ripple_host_device auto operator=(self_t&& other) -> self_t& {
    _space      = other._space;
    _data       = other._data;
    other._data = nullptr;
    return *this;
  }

  //==--- [access] ---------------------------------------------------------==//

  /// Overload of operator() to get an iterator to the element at the location
  /// specified by the \p is indices.
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  auto operator()(Indices&&... is) -> iter_t {
    return iter_t{
      allocator_t::create(_data, _space, std::forward<Indices>(is)...), _space
    };
  }

  /// Overload of operator() to get a constant iterator to the element at the
  /// location specified by the \p is indices.
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  auto operator()(Indices&&... is) const -> const_iter_t {
    return const_iter_t{
      allocator_t::create(_data, _space, std::forward<Indices>(is)...), _space
    };
  }

  //==--- [interface] ------------------------------------------------------==//

  /// Reallocates the data.
  auto reallocate() -> void {
    cleanup();
    allocate();
  }

  /// Resizes the \p dim dimension to \p dim_size.
  /// 
  /// \note This __does not__ reallocate, since multiple resizings would then
  ///       then make multiple allocations. Call reallocate to reallocate after
  ///       resizing.
  ///
  /// \param dim  The dimension to resize.
  /// \param size The size to resize the dimension to.
  /// \tparm Dim  The type of the dimension specifier.
  template <typename Dim>
  auto resize_dim(Dim&& dim, std::size_t size) -> void {
    _space[dim] = size;
  }

  /// Resizes each of the dimensions specified by the \p sizes, reallocating the
  /// data after the resizing. If less sizes are provided than there are
  /// dimensions in the block, the the first sizeof...(Sizes) dimensions will
  /// be resized.
  /// \param  sizes The sizes to resize the dimensions to.
  /// \tparam Sizes The type of the sizes.
  template <typename... Sizes>
  auto resize(Sizes&&... sizes) -> void {
    _space.resize(std::forward<Sizes>(sizes)...);
    reallocate();
  }

  /// Returns the total number of elements in the tensor.
  auto size() -> std::size_t {
    return _space.size();
  }

  /// Returns the total number of elements in dimension \p dim.
  /// \param dim The dimension to get the size of.
  /// \param Dim The type of the dimension specifier.
  template <typename Dim>
  auto size(Dim&& dim) -> std::size_t {
    return _space.size(std::forward<Dim>(dim));
  }

 private:
  ptr_t     _data = nullptr;   //!< Storage for the tensor.
  space_t   _space;            //!< Spatial information for the tensor.

  /// Allocates data for the tensor.
  auto allocate() -> void {
    if (_data == nullptr) {
      _data = malloc(allocator_t::allocation_size(_space.size()));
    }
  }

  /// Cleans up the data for the tensor.
  auto cleanup() -> void {
    if (_data != nullptr) {
      free(_data);
      _data = nullptr;
    }
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_HOST_BLOCK_HPP


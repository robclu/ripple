//==--- ripple/container/host_box.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  host_box.hpp
/// \brief This file imlements a host side box class which holds some data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_HOST_BLOCK_HPP
#define RIPPLE_CONTAINER_HOST_BLOCK_HPP

#include <ripple/iterator/block_iterator.hpp>
#include <ripple/multidim/dynamic_multidim_space.hpp>
#include <ripple/storage/storage_traits.hpp>

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


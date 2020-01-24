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

#include "block_memory_properties.hpp"
#include "block_traits.hpp"
#include "device_block.hpp"
#include <ripple/iterator/block_iterator.hpp>
#include <ripple/utility/cuda.hpp>
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
  /// Defines the type of the traits for the block.
  using traits_t        = BlockTraits<self_t>;
  /// Defines the type of allocator for the tensor.
  using allocator_t     = typename traits_t::allocator_t;
  /// Defines the the of the dimension information for the tensor.
  using space_t         = typename traits_t::space_t;
  /// Defines the type of the pointer to the data.
  using ptr_t           = void*;
  /// Defines the type for a reference to an element.
  using value_t         = typename traits_t::value_t;
  /// Defines the type of the iterator for the tensor.
  using iter_t          = BlockIterator<value_t, space_t>;
  /// Defines the type of a contant iterator.
  using const_iter_t    = const iter_t;
  /// Defines the type of a host block with the same parameters.
  using device_block_t  = DeviceBlock<T, Dimensions>;

  /// Declare device blocks to be friends, so that we can create host blocks
  /// from device blocks.
  friend device_block_t;

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

  //==--- [synchronous constuction] -----------------------------------------=//

  /// Initializes the size of each of the dimensions of the tensor, as well as
  /// the padding for the tensor. This is only enabled when the number of
  /// size arguments matches the dimensionality of the tensor, and the sizes
  /// are numeric types.
  ///
  /// \param  padding The amount of padding for the tensor.
  /// \param  sizes   The sizes of the dimensions for the tensor.
  /// \tparam Sizes   The types of other dimension sizes.
  template <
    typename... Sizes, all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  HostBlock(std::size_t padding, Sizes&&... sizes)
  : _space{padding, std::forward<Sizes>(sizes)...} {
    allocate();
  }

  /// Initializes the size of each of the dimensions of the tensor. This is only
  /// enabled when the number of arguments matches the dimensionality of the
  /// tensor, and the sizes are numeric types.
  ///
  /// \param sizes  The sizes of the each of the dimensions of the block.
  /// \tparam Sizes The types of the dimension sizes.
  template <
    typename... Sizes, all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  HostBlock(Sizes&&... sizes)
  : _space{std::forward<Sizes>(sizes)...} {
    allocate();
  }

  /// Constructor to create the block from another block.
  /// \param other The other block to create this block from.
  HostBlock(const self_t& other)
  : _space{other._space} {
    allocate();
    const auto bytes = allocator_t::allocation_size(_space.size());
    std::memcpy(_data, other._data, bytes);
  }

  /// Constructor to move the \p other block into this one. The \p other will no
  /// longer have valid data, and it's size is meaningless.
  /// \param other The other block to move to this one.
  HostBlock(self_t&& other)
  : _space{other._space} {
    _data       = other._data;
    other._data = nullptr;
  }

  /// Constructor to create the block from a device block.
  /// \param other The other block to create this block from.
  HostBlock(const device_block_t& other)
  : _space{other._space} {
    reallocate();
    cuda::memcpy_device_to_host(
      _data, other._data, allocator_t::allocation_size(_space.size())
    );
  }

  //==--- [asynchrnous construction] ---------------------------------------==//
 
  /// Initializes the size of each of the dimensions of the tensor, as well as
  /// the padding for the tensor, as well as providing an option to enable
  /// asynchronous functionality for the block. This is only enabled when the
  /// number of size arguments matches the dimensionality of the tensor, and the
  /// sizes are numeric types.
  ///
  /// \param  async   If the block must enable asynchronous functionality.
  /// \param  padding The amount of padding for the tensor.
  /// \param  sizes   The sizes of the dimensions for the tensor.
  /// \tparam Sizes   The types of other dimension sizes.
  template <
    typename... Sizes, all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  HostBlock(BlockOpKind op_kind, std::size_t padding, Sizes&&... sizes)
  : _space{padding, std::forward<Sizes>(sizes)...} {
    set_op_kind(op_kind);
    allocate();
  }

  /// Initializes the size of each of the dimensions of the tensor, as well as
  /// providing an option to enable asynchronous functionality for the block. 
  /// This is only enabled when the number of size arguments matches the
  /// dimensionality of the tensor, and the sizes are numeric types.
  ///
  /// \param  async   If the block must enable asynchronous functionality.
  /// \param  sizes   The sizes of the dimensions for the tensor.
  /// \tparam Sizes   The types of other dimension sizes.
  template <
    typename... Sizes, all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  HostBlock(BlockOpKind op_kind, Sizes&&... sizes)
  : _space{std::forward<Sizes>(sizes)...} {
    set_op_kind(op_kind);
    allocate();
  }

  //==--- [operator overloading] -------------------------------------------==//

  /// Overload of assignment operator to copy a block.
  /// \param[in] other The other block to create this block from.
  auto operator=(const self_t& other) -> self_t& {
    _space = other._space;
    reallocate();
    const auto bytes = allocator_t::allocation_size(_space.size());
    std::memcpy(_data, other._data, bytes);
    return *this;
  }

  /// Overload of assignment operator to copy a block.
  /// \param[in] other The other block to create this block from.
  auto operator=(self_t&& other) -> self_t& {
    _space      = other._space;
    _data       = other._data;
    other._data = nullptr;
    return *this;
  }

  /// Overload of assignment operator to copy a block.
  /// \param[in] other The other block to create this block from.
  auto operator=(const device_block_t& other) -> self_t& {
    _space = other._space;
    reallocate();
    cuda::memcpy_device_to_host(
      _data, other._data, allocator_t::allocation_size(_space.size())
    );
    return *this;
  }

  //==--- [conversion to device] -------------------------------------------==//

  /// Returns the host block as a device block.
  auto as_device() const -> device_block_t {
    return device_block_t{*this};
  }

  //==--- [access] ---------------------------------------------------------==//

  /// Gets an iterator to the beginning of the block.
  auto begin() -> iter_t {
    auto it = iter_t{allocator_t::create(_data, _space), _space};
    shift_iterator(it);
    return it;
  }

  /// Gets a constant iterator to the beginning of the block.
  auto begin() const -> const_iter_t {
    auto it = const_iter_t{allocator_t::create(_data, _space), _space};
    shift_iterator(it);
    return it;
  }

  /// Overload of operator() to get an iterator to the element at the location
  /// specified by the \p is indices.
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  auto operator()(Indices&&... is) -> iter_t {
    return iter_t{
      allocator_t::create(
        _data, _space, std::forward<Indices>(is) + padding()...
      ),
      _space
    };
  }

  /// Overload of operator() to get a constant iterator to the element at the
  /// location specified by the \p is indices.
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  auto operator()(Indices&&... is) const -> const_iter_t {
    return const_iter_t{
      allocator_t::create(
        _data, _space, std::forward<Indices>(is) + padding()...
      ),
      _space
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

  /// Returns the total number of elements in the block.
  auto size() const -> std::size_t {
    return _space.internal_size();
  }

  /// Returns the total number of elements in dimension \p dim.
  /// \param dim The dimension to get the size of.
  /// \param Dim The type of the dimension specifier.
  template <typename Dim>
  auto size(Dim&& dim) const -> std::size_t {
    return _space.internal_size(std::forward<Dim>(dim));
  }

  /// Returns the amount of padding for the tensor.
  auto padding() const -> std::size_t {
    return _space.padding();
  }

  /// Returns the number of dimensions for the block.
  constexpr auto dimensions() const -> std::size_t {
    return Dimensions;
  }

  /// Sets the operation kind of the block.
  /// \param op_kind The type of operation for the block.
  auto set_op_kind(BlockOpKind op_kind) -> void {
    if (op_kind == BlockOpKind::asynchronous) {
      _mem_props.pinned     = true;
      _mem_props.async_copy = true;
    } else {
      _mem_props.pinned     = false;
      _mem_props.async_copy = false;
    }
  }

  /// Sets the amount of padding for the block. This does not reallocate the
  /// memory for the block, so a call to `reallocate()` should be made if the
  /// block owns the memory.
  /// \param padding The amount of padding for the block.
  auto set_padding(size_t padding) -> void {
    _space.padding() = padding;
  }

 private:
  ptr_t            _data = nullptr;   //!< Storage for the tensor.
  space_t          _space;            //!< Spatial information for the tensor.
  BlockMemoryProps _mem_props;        //!< Memory properties for the block.

  /// Allocates data for the block, when the block only requires synchronous
  /// functionality.
  auto allocate() -> void {
    if (_data == nullptr && !_mem_props.allocated) {
      if (_mem_props.pinned) {
        cuda::allocate_host_pinned(
          reinterpret_cast<void**>(&_data),   
          allocator_t::allocation_size(_space.size())
        ); 
      } else {
        _data = malloc(allocator_t::allocation_size(_space.size()));
      }
      _mem_props.must_free = true;
      _mem_props.allocated = true;
    }
  }

  /// Cleans up the data for the tensor.
  auto cleanup() -> void {
    if (_data != nullptr && _mem_props.must_free) {
      if (_mem_props.pinned) {
        cuda::free_host_pinned(_data);
      } else {
        free(_data);
      }
      _data                 = nullptr;
      _mem_props.allocated  = false;
      _mem_props.must_free  = false;
      _mem_props.pinned     = false;
      _mem_props.async_copy = false;
    }
  }

  // Shifts an iterator by the padding in each direction.
  auto shift_iterator(iter_t& it) const -> void {
    unrolled_for<Dimensions>([&] (auto dim) {
      it.shift(dim, _space.padding());
    });
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_HOST_BLOCK_HPP


//==--- ripple/core/container/device_block.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_block.hpp
/// \brief This file imlements a device side block class which holds some data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_DEVICE_BLOCK_HPP
#define RIPPLE_CONTAINER_DEVICE_BLOCK_HPP

#include "block_traits.hpp"
#include "host_block.hpp"
#include "block_memory_properties.hpp"
#include <ripple/core/iterator/block_iterator.hpp>
#include <ripple/core/utility/cuda.hpp>

namespace ripple {

/// Implementation of a device block class which stores multidimensional data on
/// the host. This will store the data in a strided format if the type T
/// implements the StriableLayout interface and the descriptor for the storage
/// for the type has a StorageLayout::strided_view type, otherwise this will
/// store the data in a contiguous format.
///
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions>
class DeviceBlock {
  //==--- [traits] ---------------------------------------------------------==//
  /// Defines the type of this tensor.
  using self_t          = DeviceBlock<T, Dimensions>;
  /// Defines the type of the traits for the tensor.
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
  using iter_t          = typename traits_t::iter_t;
  /// Defines the type of a contant iterator.
  using const_iter_t    = const iter_t;
  /// Defines the type of a host block with the same parameters.
  using host_block_t    = HostBlock<T, Dimensions>;
  /// Defines the type of the stream for the block.
  using stream_t        = cudaStream_t;

  /// Declare host blocks to be friends, so that we can create device blocks
  /// from host blocks.
  friend host_block_t;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor which just initializes the storage. This constructor
  /// is available so that an unsized block can be instanciated and then
  /// resized at runtime.
  DeviceBlock() {
    cudaStreamCreate(&_stream);
  }

  /// Default constructor which just initializes the storage. This constructor
  /// is available so that an unsized block can be instanciated and then
  /// resized at runtime, and operations can be performed on a given stream.
  /// \param stream The stream for the device.
  DeviceBlock(stream_t stream) : _stream(stream) {}

  /// Destructor to clean up the tensor resources.
  ~DeviceBlock() {
    cleanup();
  }

  /// Initializes the size of each of the dimensions of the block. This is only
  /// enabled when the number of arguments matches the dimensionality of the
  /// block, and the sizes are numeric types.
  ///
  /// \param  sizes The sizes of the dimensions for the block.
  /// \tparam Sizes The types of other dimension sizes.
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  DeviceBlock(Sizes&&... sizes)
  : _space{std::forward<Sizes>(sizes)...} {
    cudaStreamCreate(&_stream);
    allocate();
  }

  /// Initializes the size of each of the dimensions of the tensor, as well as
  /// the padding for the tensor. This is only enabled when the number of
  /// size arguments matches the dimensionality of the tensor, and the sizes
  /// are numeric types.
  ///
  /// \param  padding The amount of padding for the tensor.
  /// \param  sizes   The sizes of the dimensions for the tensor.
  /// \tparam Sizes   The types of other dimension sizes.
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  DeviceBlock(std::size_t padding, Sizes&&... sizes)
  : _space{padding, std::forward<Sizes>(sizes)...} {
    cudaStreamCreate(&_stream);
    allocate();
  }

  /// Constructor to create the block from another block.
  /// \param other The other block to create this block from.
  DeviceBlock(const self_t& other)
  : _space{other._space}, _stream{other._stream} {
    cudaStreamCreate(&_stream);
    allocate();
    copy_from_device(other);
  }

  /// Constructor to create the block from a host block. This copies the memory
  /// from the host block into the device memory for this block.
  ///
  DeviceBlock(const host_block_t& other)
  : _space{other._space} {
    cudaStreamCreate(&_stream);
    allocate();
    copy_from_host(other);
  }

  //==--- [operator overloading] -------------------------------------------==//

  /// Constructor to create the block from another block.
  /// \param other The other block to create this block from.
  auto operator=(const self_t& other) -> self_t& {
    _stream = other._stream;
    _space  = other._space;
    reallocate();
    copy_from_device(other);
    return *this;
  }

  /// Constructor to create the block from another block.
  /// \param other The other block to create this block from.
  auto operator=(const host_block_t& other) -> self_t& {
    _space  = other._space;
    cudaStreamCreate(&_stream);
    reallocate();
    copy_from_host(other);
    return *this;
  }

  //==--- [conversion to host] ---------------------------------------------==//

  /// Returns the device block as a host block.
  auto as_host() const -> host_block_t {
    return host_block_t{*this};
  }

  //==--- [copying data] ---------------------------------------------------==//
  
  /// Copies the data from the \p other block into this block.
  /// \param other The other block to copy the data from.
  auto copy_data(const host_block_t& other) -> void {
    copy_from_host(other);
  }

  //==--- [access] ---------------------------------------------------------==//
  
  /// Gets an iterator to the beginning of the block.
  ripple_host_device auto begin() -> iter_t {
    auto it = iter_t{allocator_t::create(_data, _space), _space};
    unrolled_for<Dimensions>([&] (auto dim) {
      it.shift(dim, _space.padding());
    });
    return it;
  }

  /// Gets a constant iterator to the beginning of the block.
  ripple_host_device auto begin() const -> const_iter_t {
    auto it = const_iter_t{allocator_t::create(_data, _space), _space};
    unrolled_for<Dimensions>([&] (auto dim) {
      it.shift(dim, _space.padding());
    });
    return it;
  }

  /// Overload of operator() to get an iterator to the element at the location
  /// specified by the \p is indices.
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  ripple_host_device auto operator()(Indices&&... is) -> iter_t {
    return iter_t{
      allocator_t::create(
        _data, _space, std::forward<Indices>(is) + _space.padding()...
      ),
      _space
    };
  }

  /// Overload of operator() to get a constant iterator to the element at the
  /// location specified by the \p is indices.
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  ripple_host_device auto operator()(Indices&&... is) const -> const_iter_t {
    return const_iter_t{
      allocator_t::create(
        _data, _space, std::forward<Indices>(is) + _space.padding()...
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

  /// Returns the total number of elements in the tensor.
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

  /// Returns the number of dimensions for the block.
  constexpr auto dimensions() const -> std::size_t {
    return Dimensions;
  }

  /// Sets the amount of padding for the block. This does not reallocate the
  /// memory for the block, so a call to `reallocate()` should be made if the
  /// block owns the memory.
  /// \param padding The amount of padding.
  auto set_padding(size_t padding) -> void {
    _space.padding() = padding;
  }

  /// Returns the amount of padding for the block.
  auto padding() const -> size_t {
    return _space.padding();
  }

  /// Returns the number of bytes required to allocate the internal data for the
  /// block as well as the internal data for the block.
  auto mem_requirement() const -> size_t {
    return allocator_t::allocation_size(_space.size());
  }

  //==--- [stream] ---------------------------------------------------------==//

  /// Returns the stream used by the block.
  auto stream() const -> const stream_t& {
    return _stream;
  }

  /// Sets the stream for the block.
  /// \param stream The stream to set for the block.
  auto set_stream(const stream_t& stream) -> void {
    _stream = stream;
  }

 private:
  ptr_t            _data   = nullptr; //!< Storage for the tensor.
  space_t          _space;            //!< Spatial information for the tensor.
  stream_t         _stream;          //!< The stream to use for the block.
  BlockMemoryProps _mem_props;        //!< Memory properties for the block data.

  /// Allocates data for the tensor.
  auto allocate() -> void {
    // Can only allocate if the memory is not allocated, and if we own it.
    if (_data == nullptr && !_mem_props.allocated) {
      cuda::allocate_device(
        reinterpret_cast<void**>(&_data), mem_requirement()
      );
      _mem_props.allocated = true;
      _mem_props.must_free = true;
    }
  }

  /// Cleans up the data for the tensor.
  auto cleanup() -> void {
    if (_data != nullptr && _mem_props.must_free) {
      cuda::free_device(_data);
      _data                = nullptr;
      _mem_props.must_free = false;
      _mem_props.allocated = false;
    }
  }

  /// Copies data from the host block \p other into this block.
  /// \p other The other block to copy data from.
  auto copy_from_host(const host_block_t& other) {
    const auto alloc_size = allocator_t::allocation_size(_space.size());
    cuda::memcpy_host_to_device_async(
      _data, other._data, alloc_size, _stream
    );
  }

  /// Copies data from the host block \p other into this block.
  /// \p other The other block to copy data from.
  auto copy_from_device(const self_t& other) {
    const auto alloc_size = allocator_t::allocation_size(_space.size());
    cuda::memcpy_device_to_device_async(
      _data, other._data, alloc_size, other.stream()
    );
  }
};

//==--- [iterator extraction] ----------------------------------------------==//

/// Extracts the iterator from the \p block.
/// \param  block The block to extract the iterator from.
/// \tparam T     Type type of the data for the block.
/// \tparam Dims  The number of dimension for the block.
template <typename T, size_t Dims>
ripple_host_device auto iter_or_ref(DeviceBlock<T, Dims>& block) {
  return block.begin();
}

/// Overload of the iterator extraction function for a type which does not have
/// an iterator, returning a reference to the type which does not have an
/// iterator.
/// \param  non_block The non block type without an iterator.
/// \tparam T         The type to return a reference to.
template <typename T>
ripple_host_device auto iter_or_ref(T& non_block) -> T& {
  return non_block;
}

/// Overload of the iterator extraction function for a type which does not have
/// an iterator, returning a constant  reference to the type which does not have
/// an iterator.
/// \param  non_block The non block type without an iterator.
/// \tparam T         The type to return a reference to.
template <typename T>
ripple_host_device auto iter_or_ref(const T& non_block) -> const T& {
  return non_block;
}

} // namespace ripple

#endif // RIPPLE_CONTAINER_DEVICE_BLOCK_HPP


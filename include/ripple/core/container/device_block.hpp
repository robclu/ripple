//==--- ripple/core/container/device_block.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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
#include "../allocation/multiarch_allocator.hpp"
#include "../iterator/block_iterator.hpp"
#include "../utility/memory.hpp"

namespace ripple {

/**
 * Implementation of a device block class which stores multidimensional data on
 * the device.
 *
 * This will store the data in a strided format if the type T
 * implements the StriableLayout interface and the descriptor for the storage
 * for the type has a StorageLayout::StridedView type, otherwise this will
 * store the data in a contiguous format.
 *
 * \note While this class can be used directly, it's designed as an
 *       implementation detail for the Tensor class.
 *
 * \sa Tensor
 *
 * \tparam T          The type of the data stored in the tensor.
 * \tparam Dimensions The number of dimensions in the tensor.
 */
template <typename T, size_t Dimensions>
class DeviceBlock {
  /*==--- [traits] ---------------------------------------------------------==*/

  // clang-format off
  /** Defines the type of the traits for the block. */
  using Traits    = BlockTraits<DeviceBlock<T, Dimensions>>;
  /** Defines the type of allocator for the tensor. */
  using Allocator = typename Traits::Allocator;
  /** Defines the the of the dimension information for the tensor. */
  using Space     = typename Traits::Space;
  /** Defines the type of the pointer to the data. */
  using Ptr       = void*;
  /** Defines the type for a reference to an element. */
  using Value     = typename Traits::Value;
  /** Defines the type of the iterator for the tensor.  */
  using Iter      = typename Traits::Iter;
  /** Defines the type of a constant iterator. */
  using ConstIter = const Iter;
  /** Defines the type of a host block with the same parameters. */
  using HostBlock = HostBlock<T, Dimensions>;
  /** Defines the type of the stream for the block. */
  using Stream    = GpuStream;
  // clang-format on

  /**
   * Declare host blocks to be friends so for copy construction.
   */
  friend HostBlock;

  /**
   * Declare Block a friend to allow it to allow it to set properties of the
   * device block.
   */
  template <typename Type, size_t Dims>
  friend class Block;

 public:
  // clang-format off
  /** Defines the type used for padding. */
  using Padding      = typename Space::Padding;
  /** Defines the type of the pointer to the allocator. */
  using AllocatorPtr = MultiarchAllocator*;
  // clang-format on

  /**
   * Swaps the blocks lhs and rhs blocks.
   * \param lhs The left block to swap.
   * \param rhs The right block to swap.
   */
  friend auto swap(DeviceBlock& lhs, DeviceBlock& rhs) noexcept -> void {
    using std::swap;
    swap(lhs.data_, rhs.data_);
    swap(lhs.allocator_, rhs.allocator_);
    swap(lhs.mem_props_, rhs.mem_props_);
    swap(lhs.space_, rhs.space_);
  }

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor.This constructor is available so that an unsized block
   * can be instanciated and then resized at runtime.
   */
  DeviceBlock() noexcept {
    gpu::create_stream(&stream_);
  }

  /**
   * Constructor which sets the stream for the block.
   * \param stream The stream for the device.
   */
  DeviceBlock(Stream stream, AllocatorPtr allocator = nullptr) noexcept
  : allocator_{allocator}, stream_{stream} {}

  /**
   * Default constructor.This constructor is available so that an unsized block
   * can be instanciated and then resized at runtime.
   */
  DeviceBlock(AllocatorPtr allocator) noexcept : allocator_{allocator} {
    gpu::create_stream(&stream_);
  }

  /**
   * Constructor which sets the stream for the block.
   * \param stream The stream for the device.
   */
  DeviceBlock(Stream stream) noexcept : stream_{stream} {}

  /**
   * Destructor for the block, which cleans up the block resources.
   */
  ~DeviceBlock() noexcept {
    cleanup();
  }

  /**
   * Initializes the size of each of the dimensions of the block.
   *
   * \note This is only enabled when the number of arguments matches the
   *       dimensionality of the block and the sizes are arithmetic types.
   *
   * \note This also allocated the data for the block.
   *
   * \param  sizes The sizes of the dimensions for the block.
   * \tparam Sizes The types of other dimension sizes.
   */
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0>
  DeviceBlock(Sizes&&... sizes) : space_{ripple_forward(sizes)...} {
    gpu::create_stream(&stream_);
    allocate();
  }

  /**
   * Initializes the size of each of the dimensions of the block, as well as
   * the padding for the block.
   *
   * \note This is only enabled when the number of size arguments matches the
   *       dimensionality of the block and the sizes are arithmetic types.
   *
   * \note This also allocates data for the block.
   *
   * \note The amount of padding specified adds the given amount to each side
   *       of each dimension.
   *
   * \param  padding The amount of padding for the block.
   * \param  sizes   The sizes of the dimensions for the block.
   * \tparam Sizes   The types of other dimension sizes.
   */
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0>
  DeviceBlock(Padding padding, Sizes&&... sizes)
  : space_{padding, ripple_forward(sizes)...} {
    gpu::create_stream(&stream_);
    allocate();
  }

  /**
   * Copy constructor to create the block from another block.
   * \param other The other block to create this block from.
   */
  DeviceBlock(const DeviceBlock& other)
  : space_{other.space_}, stream_{other.stream_} {
    gpu::create_stream(&stream_);
    allocate();
    copy_from_device(other);
  }

  /**
   * Constructor to move the other block into this one.
   * The other block will no longer have valid data, and it's size is
   * meaningless.
   * \param other The other block to move to this one.
   */
  DeviceBlock(DeviceBlock&& other) noexcept : space_{other.space_} {
    data_       = other.data_;
    device_id_  = other.device_id_;
    stream_     = other.stream_;
    other.data_ = nullptr;
    mem_props_  = other.mem_props_;
    other.mem_props_.reset();
  }

  /**
   * Constructor to create the block from a host block.
   *
   * \note This copies the memory from the host block into the device memory for
   *       this block.
   *
   * \param other The host block to create this block from.
   */
  DeviceBlock(const HostBlock& other)
  : allocator_{other.allocator_}, space_{other.space_} {
    gpu::create_stream(&stream_);
    allocate();
    copy_from_host(other);
  }

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Overload of copy assignment to create the block from another block.
   *
   * \note This copies the data from the other block into this one.
   *
   * \param other The other block to create this block from.
   * \return A reference to the created block.
   */
  auto operator=(const DeviceBlock& other) -> DeviceBlock& {
    stream_ = other.stream_;
    space_  = other.space_;
    reallocate();
    copy_from_device(other);
    return *this;
  }

  /**
   * Overload of copy assignment to create the block from another block.
   *
   * \note This copies the data from the other block into this one.
   *
   * \param other The other block to create this block from.
   * \return A reference to the created block.
   */
  auto operator=(const HostBlock& other) -> DeviceBlock& {
    space_     = other._space;
    allocator_ = other.allocator_;
    gpu::create_stream(&stream_);
    reallocate();
    copy_from_host(other);
    return *this;
  }

  /**
   * Overload of operator() to get an iterator to the element at the location
   * specified by the indices.
   *
   * \param  is      The indices to get the element at.
   * \tparam Indices The types of the indices.
   * \return An iterator pointing to the element defined by the indices.
   */
  template <typename... Indices>
  ripple_host_device auto operator()(Indices&&... is) noexcept -> Iter {
    return Iter{
      Allocator::create(
        data_, space_, ripple_forward(is) + space_.padding()...),
      space_};
  }

  /**
   * Overload of operator() to get a constant iterator to the element at the
   * location specified by the indices.
   *
   * \param  is      The indices to get the element at.
   * \tparam Indices The types of the indices.
   * \return A const iterator pointing to the element defined by the indices.
   */
  template <typename... Indices>
  ripple_host_device auto
  operator()(Indices&&... is) const noexcept -> ConstIter {
    return ConstIter{
      Allocator::create(
        data_, space_, ripple_forward(is) + space_.padding()...),
      space_};
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Create a host block with the data from this device block.
   * \param op_kind The kind of the memory operations for the block.
   * \return A host block with the data and properties of this block.
   */
  auto
  as_host(BlockOpKind op_kind = BlockOpKind::synchronous) const -> HostBlock {
    return HostBlock{*this, op_kind};
  }

  /**
   * Copies the data from the host block into this block.
   * \param other The other block to copy the data from.
   */
  auto copy_data(const HostBlock& other) noexcept -> void {
    copy_from_host(other);
  }

  /**
   * Gets an iterator to the beginning of the block.
   *
   * \note The returned iterator points to the first valid cell (i.e is is
   *       offset over the padding).
   *
   * \return An iterator to the first element in the block.
   */
  ripple_host_device auto begin() noexcept -> Iter {
    auto it = Iter{Allocator::create(data_, space_), space_};
    unrolled_for<Dimensions>(
      [&](auto dim) { it.shift(dim, space_.padding()); });
    return it;
  }

  /**
   * Gets a constant iterator to the beginning of the block.
   *
   * \note The returned iterator points to the first valid cell (i.e is is
   *       offset over the padding).
   *
   * \return A constant iterator to the first element in the block.
   */
  ripple_host_device auto begin() const noexcept -> ConstIter {
    auto it = ConstIter{Allocator::create(data_, space_), space_};
    unrolled_for<Dimensions>(
      [&](auto dim) { it.shift(dim, space_.padding()); });
    return it;
  }

  /**
   * Reallocates the data for the block.
   */
  auto reallocate() -> void {
    cleanup();
    allocate();
  }

  /**
   * Resizes the dimension to the given size.
   *
   * \note This *does not* reallocate, since multiple resizings would then
   *       then make multiple allocations. Call reallocate to reallocate after
   *       resizing.
   *
   * \note The resize *does not* include padding, padding for the dimension is
   *       *in addition to* this size.
   *
   * \param dim  The dimension to resize.
   * \param size The size to resize the dimension to.
   * \tparm Dim  The type of the dimension specifier.
   */
  template <typename Dim>
  auto resize_dim(Dim&& dim, size_t size) noexcept -> void {
    space_[dim] = size;
  }

  /**
   * Resizes each of the dimensions specified to the given size.
   *
   * \note This *does* reallocate the data after the resize, since it is assumed
   *       that no further resizing is necessary since all dimension sizes are
   *       specified in this call.
   *
   * \note If less sizes are provided than there are dimensions in the block,
   *       the the first sizeof...(Sizes) dimensions will be resized.
   *
   * \param  sizes The sizes to resize the dimensions to.
   * \tparam Sizes The type of the sizes.
   */
  template <typename... Sizes>
  auto resize(Sizes&&... sizes) -> void {
    space_.resize(ripple_forward(sizes)...);
    reallocate();
  }

  /**
   * Gets the total number of * valid* elements in the block, therefore this
   * size *excludes* padding elements.
   *
   * \sa pitch
   *
   * \return The total number of valid elements in the block.
   */
  auto size() const noexcept -> size_t {
    return space_.internal_size();
  }

  /**
   * Gets the total number of *valid* elements in the dimension, therefore this
   * *excludes* the padding elements for the dimension.
   *
   * \sa pitch
   *
   * \param dim The dimension to get the size of.
   * \param Dim The type of the dimension specifier.
   * \return The number of valid elements in the dimension.
   */
  template <typename Dim>
  auto size(Dim&& dim) const -> size_t {
    return space_.internal_size(ripple_forward(dim));
  }

  /**
   * Gets the pitch of the block in the given dimension, which is the
   * total number of elements in the dimension *including* padding.
   *
   * \param  dim The dimension to get the pitch for.
   * \tparam Dim The type of the dimension specifier.
   * \return The total number of elements for the dimension *including* padding
   *         elements.
   */
  template <typename Dim>
  constexpr auto pitch(Dim&& dim) const noexcept -> size_t {
    return space_.size(ripple_forward(dim));
  }

  /**
   * Gets the number of dimensions for the block.
   * \return The number of dimensions for the block.
   */
  constexpr auto dimensions() const noexcept -> size_t {
    return Dimensions;
  }

  /**
   * Sets the amount of padding for a single side of a single dimension for the
   * block. This padding amount will be applied to all sides of all dimensions.
   *
   * \note This does not reallocate the memory for the block, so a call to
   *       reallocate should be made when the data should be reallocated.
   *
   * \param padding The amount of padding.
   */
  auto set_padding(Padding padding) noexcept -> void {
    space_.padding() = padding;
  }

  /**
   * Gets the amount of padding for the block.
   *
   * \note This padding amount is for a single side of a single dimension of
   *       the block.
   *
   * \return The amount of padding for the block.
   */
  auto padding() const noexcept -> Padding {
    return space_.padding();
  }

  /**
   * Gets the number of bytes required to allocate memory to hold all data
   * for the block.
   * \return The number of bytes required for the block.
   */
  auto mem_requirement() const noexcept -> size_t {
    return Allocator::allocation_size(space_.size());
  }

  /**
   * Sets the device id for the block.
   * \param device_id The id of the device to use for the block.
   */
  auto set_device_id(uint32_t device_id) noexcept -> void {
    device_id_ = device_id;
  }

  /**
   * Gets the device id for the block.
   * \return The device id for this block.
   */
  auto device_id() const noexcept -> uint32_t {
    return device_id_;
  }

  /**
   * Gets the stream used by the block.
   * \return The stream for the block.
   */
  auto stream() const noexcept -> Stream {
    return stream_;
  }

  /**
   * Sets the stream for the block.
   * \param stream The stream to set for the block.
   */
  auto set_stream(const Stream& stream) noexcept -> void {
    stream_ = stream;
  }

  /**
   * Destroys the stream for the block.
   */
  auto destroy_stream() noexcept -> void {
    gpu::set_device(device_id_);
    gpu::destroy_stream(stream_);
  }

  /**
   * Returns the copy type required to copy from this block to another block
   * with type Block.
   * \tparam Block The block to determine the copy type from.
   * \return The type of the copy required to copy from this block.
   */
  template <typename Block>
  constexpr auto get_copy_type() const noexcept -> cudaMemcpyKind {
    return is_host_block_v<Block> ? cudaMemcpyDeviceToHost
                                  : cudaMemcpyDeviceToDevice;
  }

 private:
  Ptr              data_      = nullptr; //!< Storage for the tensor.
  AllocatorPtr     allocator_ = nullptr; //!< Allocator for device data.
  Space            space_;         //!< Spatial information for the tensor.
  Stream           stream_;        //!< The stream to use for the block.
  uint32_t         device_id_ = 0; //!< Id of the device for the block.
  BlockMemoryProps mem_props_;     //!< Memory properties for the block data.

  /**
   * Allocates data for the block.
   */
  auto allocate() -> void {
    // Can only allocate if the memory is not allocated, and if we own it.
    if (data_ == nullptr && !mem_props_.allocated) {
      gpu::set_device(device_id_);
      if (allocator_ != nullptr) {
        data_ = allocator_->gpu_allocator(device_id_)
                  .alloc(mem_requirement(), Traits::alignment);
      } else {
        gpu::allocate_device(
          reinterpret_cast<void**>(&data_), mem_requirement());
      }
      mem_props_.allocated = true;
      mem_props_.must_free = true;
    }
  }

  /**
   * Cleans up the data for the tensor.
   */
  auto cleanup() -> void {
    if (data_ != nullptr && mem_props_.must_free) {
      gpu::set_device(device_id_);
      if (allocator_ != nullptr) {
        allocator_->gpu_allocator(device_id_).free(data_);
      } else {
        gpu::free_device(data_);
      }
      data_                = nullptr;
      mem_props_.must_free = false;
      mem_props_.allocated = false;
    }
  }

  /**
   * Copies data from the host block into this block.
   * \param other The other block to copy data from.
   */
  auto copy_from_host(const HostBlock& other) noexcept -> void {
    const auto alloc_size = Allocator::allocation_size(space_.size());
    gpu::set_device(device_id_);
    gpu::memcpy_host_to_device_async(data_, other.data_, alloc_size, stream_);
  }

  /**
   * Copies data from the device block into this one.
   * \param other The other block to copy data from.
   */
  auto copy_from_device(const DeviceBlock& other) noexcept -> void {
    const auto alloc_size = Allocator::allocation_size(space_.size());
    gpu::set_device(device_id_);
    gpu::memcpy_device_to_device_async(
      data_, other.data_, alloc_size, other.stream());
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_DEVICE_BLOCK_HPP

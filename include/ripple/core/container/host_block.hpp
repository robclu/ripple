//==--- ripple/core/container/host_block.hpp -------------------- -*- C++ -*-
//---==//
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
#include <ripple/core/iterator/block_iterator.hpp>
#include <ripple/core/utility/cuda.hpp>
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
template <typename T, size_t Dimensions>
class HostBlock {
  //==--- [traits] ---------------------------------------------------------==//
  // clang-format off
  /** Defines the type of the traits for the block. */
  using Traits      = BlockTraits<DeviceBlock<T, Dimensions>>;
  /** Defines the type of allocator for the tensor. */
  using Allocator   = typename Traits::Allocator;
  /** Defines the the of the dimension information for the tensor. */
  using Space       = typename Traits::Space;
  /** Defines the type of the pointer to the data. */
  using Ptr         = void*;
  /** Defines the type for a reference to an element. */
  using Value       = typename Traits::Value;
  /** Defines the type of the iterator for the tensor.  */
  using Iter        = typename Traits::Iter;
  /** Defines the type of a constant iterator. */
  using ConstIter   = const Iter;
  /** Defines the type of a host block with the same parameters. */
  using DeviceBlock = DeviceBlock<T, Dimensions>;
  /** Defines the type of the stream for the block. */
  using Stream      = cudaStream_t;
  // clang-format on

  /**
   * Declare device blocks to be friends so for copy construction.
   */
  friend DeviceBlock;

  /**
   * Declare Block a friend to allow it to allow it to set properties of the
   * host block.
   */
  template <typename Type, size_t Dims>
  friend class Block;

 public:
  /** Defines the type of the padding. */
  using Padding = typename Space::Padding;

  /**
   * Swaps the blocks lhs and rhs blocks.
   * \param lhs The left block to swap.
   * \param rhs The right block to swap.
   */
  friend auto swap(HostBlock& lhs, HostBlock& rhs) noexcept -> void {
    using std::swap;
    swap(lhs.data_, rhs.data_);
  }

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor.This constructor is available so that an unsized block
   * can be instanciated and then resized at runtime.
   */
  HostBlock() = default;

  /**
   * Destructor for the block, which cleans up the block resources.
   */
  ~HostBlock() noexcept {
    cleanup();
  }

  //==--- [synchronous constuction] -----------------------------------------=//

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
  HostBlock(Sizes&&... sizes) noexcept
  : space_{static_cast<Sizes&&>(sizes)...} {
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
  HostBlock(Padding padding, Sizes&&... sizes)
  : space_{padding, static_cast<Sizes&&>(sizes)...} {
    allocate();
  }

  /**
   * Copy constructor to create the block from another block.
   * \param other The other block to create this block from.
   */
  HostBlock(const HostBlock& other) : space_{other.space_} {
    const auto bytes = Allocator::allocation_size(space_.size());
    mem_props_       = other.mem_props_;
    std::memcpy(data_, other.data_, bytes);
  }

  /**
   * Constructor to move the other block into this one.
   * The other block will no longer have valid data, and it's size is
   * meaningless.
   * \param other The other block to move to this one.
   */
  HostBlock(HostBlock&& other) noexcept : space_{other.space_} {
    data_       = other.data_;
    other.data_ = nullptr;
    mem_props_  = other.mem_props_;
    other.mem_props_.reset();
  }

  /**
   * Constructor to create the block from a device block.
   * \param other The other block to create this block from.
   */
  HostBlock(const DeviceBlock& other) : space_{other.space_} {
    reallocate();
    copy_from_device(other);
  }

  /*==--- [asynchrnous construction] ---------------------------------------==*/

  /**
   * Initializes the size of each of the dimensions of the block, as well as
   * providing an option to enable asynchronous functionality for the block.
   *
   * \note This is only enabled when the number of size arguments matches the
   *       dimensionality of the tensor, and the sizes are arithmetic types.
   *
   *
   * \param  op_kind The kind of operation for the block.
   * \param  sizes   The sizes of the dimensions for the tensor.
   * \tparam Sizes   The types of other dimension sizes.
   */
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0>
  HostBlock(BlockOpKind op_kind, Sizes&&... sizes)
  : space_{static_cast<Sizes&&>(sizes)...} {
    set_op_kind(op_kind);
    allocate();
  }

  /**
   * Initializes the size of each of the dimensions of the block, the padding
   * for the block, and provides an option to enable asynchronous functionality
   * for the block.
   *
   * \note This is only enabled when the number of size arguments matches the
   *       dimensionality of the block, and the izes are arithmetic types.
   *
   * \param  op_kind The operation kind for the block.
   * \param  padding The amount of padding for the tensor.
   * \param  sizes   The sizes of the dimensions for the tensor.
   * \tparam Sizes   The types of other dimension sizes.
   */
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0>
  HostBlock(BlockOpKind op_kind, size_t padding, Sizes&&... sizes)
  : space_{padding, static_cast<Sizes&&>(sizes)...} {
    set_op_kind(op_kind);
    allocate();
  }

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Overload of copy assignment operator to create this block from another.
   * \param other The other block to create this block from.
   * \return A reference to the created block.
   */
  auto operator=(const HostBlock& other) -> HostBlock& {
    space_ = other.space_;
    reallocate();
    const auto bytes = Allocator::allocation_size(space_.size());
    std::memcpy(data_, other.data_, bytes);
    return *this;
  }

  /**
   * Overload of move assignment operator to move another block into this block.
   * \param other The other block to create this block from.
   * \return A reference to the created block.
   */
  auto operator=(HostBlock&& other) noexcept -> HostBlock& {
    space_      = other.space_;
    data_       = other.data_;
    other.data_ = nullptr;
    mem_props_  = other.mem_props_;
    other.mem_props_.reset();
    return *this;
  }

  /**
   * Overload of copy assignment operator to create this block from the other
   * block.
   * \param other The other block to create this block from.
   * \return A reference to the created block.
   */
  auto operator=(const DeviceBlock& other) -> HostBlock& {
    space_ = other.space_;
    reallocate();
    copy_from_device(other);
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
  auto operator()(Indices&&... is) noexcept -> Iter {
    return Iter{Allocator::create(
                  data_, space_, static_cast<Indices&&>(is) + padding()...),
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
  auto operator()(Indices&&... is) const noexcept -> ConstIter {
    return ConstIter{
      Allocator::create(
        data_, space_, static_cast<Indices&&>(is) + padding()...),
      space_};
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Creates a device block with the data from this host block.
   * \return A device block with the data and properties of this block.
   */
  auto as_device() const -> DeviceBlock {
    return DeviceBlock{*this};
  }

  /**
   * Copies the data from the device block into this block.
   * \param other The other block to copy the data from.
   */
  auto copy_data(const DeviceBlock& other) noexcept -> void {
    copy_from_device(other);
  }

  /**
   * Gets an iterator to the beginning of the block.
   *
   * \note The returned iterator points to the first valid cell (i.e is is
   *       offset over the padding).
   *
   * \return An iterator to the first element in the block.
   */
  auto begin() noexcept -> Iter {
    auto it = Iter{Allocator::create(data_, space_), space_};
    shift_iterator(it);
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
  auto begin() const -> ConstIter {
    auto it = ConstIter{Allocator::create(data_, space_), space_};
    shift_iterator(it);
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
   * Reallocates the data and initializes the data using the givenargs for the
   * construction of the elements.
   * \param  args The arguments for the constructor.
   * \tparam Args The type of the arguments.
   */
  template <typename... Args>
  auto reallocate_and_init(Args&&... args) -> void {
    cleanup();
    allocate_and_init(static_cast<Args&&>(args)...);
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
    space_.resize(static_cast<Sizes&&>(sizes)...);
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
  auto size(Dim&& dim) const noexcept -> size_t {
    return space_.internal_size(static_cast<Dim&&>(dim));
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
    return space_.size(static_cast<Dim&&>(dim));
  }

  /**
   * Gets the number of dimensions for the block.
   * \return The number of dimensions for the block.
   */
  constexpr auto dimensions() const noexcept -> size_t {
    return Dimensions;
  }

  /**
   * Sets the operation kind of the block to either synchronous or asynchronous.
   * \param op_kind The type of operation for the block.
   */
  auto set_op_kind(BlockOpKind op_kind) -> void {
    if (op_kind == BlockOpKind::asynchronous) {
      mem_props_.pinned     = true;
      mem_props_.async_copy = true;
      return;
    }
    mem_props_.pinned     = false;
    mem_props_.async_copy = false;
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
   * Returns the stream used by the block.
   * \return The stream for the block.
   */
  auto stream() const -> cudaStream_t {
    return 0;
  }

  /**
   * Returns the copy type required to copy from this block to another block
   * with type Block.
   * \tparam Block The block to determine the copy type from.
   * \return The type of the copy required to copy from this block.
   */
  template <typename Block>
  constexpr auto get_copy_type() const -> cudaMemcpyKind {
    return is_host_block_v<Block> ? cudaMemcpyHostToHost
                                  : cudaMemcpyHostToDevice;
  }

 private:
  Ptr              data_ = nullptr; //!< Storage for the tensor.
  Space            space_;          //!< Spatial information for the tensor.
  BlockMemoryProps mem_props_;      //!< Memory properties for the block.

  /**
   * Allocates data for the block.
   */
  auto allocate() -> void {
    if (data_ == nullptr && !mem_props_.allocated) {
      if (mem_props_.pinned) {
        cuda::allocate_host_pinned(
          reinterpret_cast<void**>(&data_), mem_requirement());
      } else {
        data_ = malloc(mem_requirement());
      }
      mem_props_.must_free = true;
      mem_props_.allocated = true;
    }
  }

  /**
   * Allocates data for the block, and then initializes the elements by
   * forwarding the args to the constructor.
   * \param  args The args for the constructor.
   * \tparam Args The types of the args.
   */
  template <typename... Args>
  auto allocate_and_init(Args&&... args) -> void {
    allocate();
    invoke(
      *this,
      [](auto&& it, auto&&... as) {
        using ItType = std::decay_t<decltype(*it)>;
        new (&(*it)) ItType{static_cast<decltype(as)&&>(as)...};
      },
      static_cast<Args&&>(args)...);
  }

  /**
   * Cleans up the data for the block.
   */
  auto cleanup() -> void {
    if (data_ != nullptr && mem_props_.must_free && mem_props_.allocated) {
      if (mem_props_.pinned) {
        cuda::free_host_pinned(data_);
      } else {
        free(data_);
      }
      data_                = nullptr;
      mem_props_.allocated = false;
      mem_props_.must_free = false;
    }
  }

  /**
   * Copies data from the device block into this one.
   * \param other The other block to copy data from.
   */
  auto copy_from_device(const DeviceBlock& other) noexcept -> void {
    const auto alloc_size = Allocator::allocation_size(space_.size());
    cuda::memcpy_device_to_host_async(
      data_, other.data_, alloc_size, other.stream());
  }

  /**
   * Shifts an iterator by the padding in each direction.
   */
  auto shift_iterator(Iter& it) const noexcept -> void {
    unrolled_for<Dimensions>(
      [&](auto dim) { it.shift(dim, space_.padding()); });
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_HOST_BLOCK_HPP

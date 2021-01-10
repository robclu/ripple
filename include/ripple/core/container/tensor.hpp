//==--- ripple/core/container/tensor.hpp ------------------- -*- C++ -*- ---==//
//
//                                  Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor.hpp
/// \brief This file defines a class which represents a tensor -- An N
///        dimensional space across multiple devices.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_TENSOR_HPP
#define RIPPLE_CONTAINER_TENSOR_HPP

#include "block.hpp"
#include "tensor_traits.hpp"
#include <ripple/core/arch/topology.hpp>
#include <ripple/core/functional/invoke.hpp>
#include <ripple/core/math/math.hpp>
#include <ripple/core/multidim/dynamic_multidim_space.hpp>
#include <ripple/core/utility/dim.hpp>
#include <array>
#include <cassert>
#include <numeric>
#include <set>

namespace ripple {

struct BlockExtractor;

/**
 * Implementation of a tensor class for an N-dimensional container for which
 * the data resides on both the host and the device.
 *
 * \tparam T          The type of the data for the grid.
 * \tparam Dimensions The number of dimensions for the grid
 */
template <typename T, size_t Dimensions>
class Tensor {
  /** Allow the splitter access to partition the work. */
  friend struct BlockExtractor;

  // clang-format off
  /** Defines the type of the block for the grid. */
  using Block         = Block<T, Dimensions>;
  /** Defines the type of the container used for the blocks. */
  using Blocks        = HostBlock<Block, Dimensions>;
  /** Defines the type of the host block for the tensor. */
  using HostBlock     = HostBlock<T, Dimensions>;
  /** Defines the type of the device block for the tensor. */
  using DeviceBlock   = DeviceBlock<T, Dimensions>;
  /** Defines the type of the space used by the grid. */
  using Space         = DynamicMultidimSpace<Dimensions>;
  /** Defines the type of teh host block iterator. */
  using HostBlockIter = IndexedIterator<T, Space>;

  /** Defines the min threshold for which splitting to multiple gpus occurs. */
  static constexpr size_t   min_split_threshold          = 3e5;
  /** Defines the number of dimensions for the tensor. */
  static constexpr size_t   dims                         = Dimensions;
  /** Default number of blocks per partition. */
  static constexpr uint32_t default_blocks_per_partition = 1;

 public:
  /** Defines the size type used. */
  using Size       = uint32_t;
  /** Defines the type of the block split specifier. */
  using BlockSplit = std::array<Size, dims>;
  /** Defines the type to store partition information. */
  using Partitions = std::array<Size, dims>;
  /** Container to hold a number of elements. */
  using Elements   = std::array<Size, dims>;
  /** Defines the type of an iterator over the blocks. */
  using BlockIter  = typename block_traits_t<Blocks>::Iter;
  // clang-format on

  /*==--- [friends] --------------------------------------------------------==*/

  /**
   * Swaps the two tensors.
   *
   * \note this goes through each block and swaps the data for each block.
   *
   * \param lhs The left tensor to swap.
   * \param rhs The right tensor to swap.
   */
  friend auto swap(Tensor& lhs, Tensor& rhs) noexcept -> void {
    invoke_generic(
      CpuExecutor(),
      [&](auto&& left_it, auto&& right_it) {
        using std::swap;
        swap(left_it->host_data, right_it->host_data);
        swap(left_it->device_data, right_it->device_data);

        auto stream = left_it->stream();
        left_it->device_data.set_stream(right_it->stream());
        right_it->device_data.set_stream(stream);
      },
      lhs.blocks_,
      rhs.blocks_);
  }

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor. This does not initialize or allocate any data for the
   * tensor. This constructor allows the tensor to be sized dynamically and then
   * allocated, for cases where the sizes of the dimensions are not know
   * beforehand.
   */
  Tensor() noexcept {}

  /**
   * Contructor to intialize the tensor from only the dimension sizes.
   * This will parition across the largest dimension.
   * \param sizes The sizes of the dimensions for the tensor.
   */
  template <typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> = 0>
  Tensor(Sizes&&... sizes)
  : space_{static_cast<Sizes&&>(sizes)...}, blocks_per_part_set_{true} {
    blocks_per_part_.fill(default_blocks_per_partition);
    partitions_.fill(1);
    size_t max_dim = 0, max_size = 0, dim = 0;
    for_each(
      [&max_dim, &dim, &max_size](auto&& dim_size) {
        if (dim_size > max_size) {
          max_size = dim_size;
          max_dim  = dim;
        }
        dim++;
      },
      static_cast<Sizes&&>(sizes)...);
    partitions_[max_dim] = topology().num_gpus();
    initialize();
  }

  /**
   * Constuctor to create the tensor with size specification and number of
   * partitions per dimension.
   *
   * \note This is only enabled if the number of dimension sizes match the
   *       number of dimensions.
   *
   * \param  partitions_per_dim Partitions pus per dimension.
   * \param  sizes              The sizes for each dimension.
   * \tparam Sizes              The types of the sizes of the dimensions.
   */
  template <typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> = 0>
  Tensor(Partitions partitions_per_dim, Sizes&&... sizes)
  : space_{static_cast<Sizes&&>(sizes)...},
    partitions_{partitions_per_dim},
    blocks_per_part_set_{true} {
    blocks_per_part_.fill(default_blocks_per_partition);
    initialize();
  }

  /**
   * Constuctor to create the tensor with a size specification, the number of
   * partitiosn per dimension, and a padding amount.
   *
   * \note This is only enabled if the number of dimension sizes match the
   *       number of dimensions.
   *
   * \param  partitions_per_dim Partitions pus per dimension.
   * \param  padding       The amount of padding for the tensor.
   * \param  sizes         The sizes for each dimension.
   * \tparam Sizes         The types of the sizes of the dimensions.
   */
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0>
  Tensor(Partitions partitions_per_dim, uint32_t padding, Sizes&&... sizes)
  : space_{padding, static_cast<Sizes&&>(sizes)...},
    partitions_{partitions_per_dim},
    blocks_per_part_set_{true} {
    blocks_per_part_.fill(default_blocks_per_partition);
    initialize();
  }

  /**
   * Constuctor to create the tensor with size specification, the number of
   * partitions per dimension, and the number of blocks per partition.
   *
   * \note This is only enabled if the number of dimension sizes match the
   *       number of dimensions.
   *
   * \param  gpus_per_partition   The number of partitions per dimension.
   * \param  blocks_per_partition Blocks per partition for each dimension.
   * \param  sizes                The sizes for each dimension.
   * \tparam Sizes                The types of the sizes of the dimensions.
   */
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0>
  Tensor(
    Partitions partitions_per_dim,
    BlockSplit blocks_per_partition,
    Sizes&&... sizes)
  : space_{static_cast<Sizes&&>(sizes)...},
    partitions_{partitions_per_dim},
    blocks_per_part_{blocks_per_partition},
    blocks_per_part_set_{true} {
    initialize();
  }

  /**
   * Constuctor to create the tensor with size specification, the number of gpus
   * per dimension, and the number of blocks per gpu, and an amount of padding
   * for the tensor.
   *
   * \note This is only enabled if the number of dimension sizes match the
   *       number of dimensions.
   *
   * \param  gpus_per_partition   The number of partitions per dimension.
   * \param  blocks_per_partition Blocks per partition for each dimension
   * \param  padding              The amount of padding for the tensor.
   * \param  sizes                The sizes for each dimension.
   * \tparam Sizes                The types of the sizes of the dimensions.
   */
  template <
    typename... Sizes,
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0>
  Tensor(
    Partitions partitions_per_dim,
    BlockSplit blocks_per_partition,
    uint32_t   padding,
    Sizes&&... sizes)
  : space_{padding, static_cast<Sizes&&>(sizes)...},
    partitions_{partitions_per_dim},
    blocks_per_part_{blocks_per_partition},
    blocks_per_part_set_{true} {
    initialize();
  }

  /*===--- [operator overloads] --------------------------------------------==*/

  /**
   * Gets a host iterator to an element at the given location.
   *
   * \note Since the tensor data is divided into blocks, the iterator is only
   *       valid for the block in which the elements exists.
   *
   * \sa IndexedIterator, BlockIterator
   *
   * \note This may be a slow operation if the data is on the device, since the
   *       block data will then need to be copied to the host before returning
   *       the value.
   *
   * \note This function is provided mainly for validation of tensor data.
   *
   * \param  indices The indices of the element to get.
   * \tparam Indices The types of teh indices.
   * \return An iterator to the block in which the element resides, pointing to
   *         the element.
   */
  template <typename... Indices>
  auto operator()(Indices&&... indices) const noexcept -> HostBlockIter {
    std::array<int, dims> ids        = {static_cast<int>(indices)...};
    auto                  block_iter = blocks_.begin();
    unrolled_for<dims>([&](auto dim) {
      const int id = ids[dim] / block_sizes_[dim];
      block_iter.shift(dim, id);
    });
    block_iter->ensure_host_data_available();
    block_iter->synchronize();

    auto host_iter = block_iter->host_iterator();
    unrolled_for<dims>([&](auto dim) {
      const int offset = ids[dim] % block_sizes_[dim];
      host_iter.shift(dim, offset);
    });
    return host_iter;
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets the size of the tensor in the given dimension. This size is the number
   * of elements in the dimension *excluding* padding.
   *
   * \sa pitch
   *
   * \param  dim The dimension to get the size of.
   * \tparam Dim The type of the dimension specifier.
   * \return The number of elements for the given dimension.
   */
  template <typename Dim>
  auto size(Dim&& dim) const noexcept -> size_t {
    return space_.internal_size(static_cast<Dim&&>(dim));
  }

  /**
   * Gets the pitch of the tensor in the given dimension. This pitch is the
   * number of elements in the dimension *including* padding.
   *
   * \sa size
   *
   * \param  dim The dimension to get the pitch of.
   * \tparam Dim The type of the dimension specifier.
   * \return The number of elements for the given dimension including padding.
   */
  template <typename Dim>
  auto pitch(Dim&& dim) const noexcept -> size_t {
    return space_.size(static_cast<Dim&&>(dim));
  }

  /**
   * Gets the amount of padding for the tensor.
   * \return The amount of padding for the tensor.
   */
  auto padding() const noexcept -> size_t {
    return space_.padding();
  }

  /**
   * Sets the number of partitions for the given dimension. This will divide the
   * tensor in the dimension into the given number of partitions.
   *
   * \param  dim        The dimension to set the number of partitions for.
   * \param  partitions The number of partitions for the dimension.
   * \tparam Dim        The dimension type.
   */
  template <typename Dim>
  auto set_partitions(Dim&& dim, size_t partitions) noexcept -> void {
    partitions_[dim] = partitions;
  }

  /**
   * Sets the number of sub partitions for the dimension, which is the number
   * of blocks per partitions for the dimension.
   * \param  dim            The dimension to set the sub partitions for.
   * \param  sub_partitions The number of sub partitions.
   * \tparam Dim            The type of the dimension specifier.
   */
  template <typename Dim>
  auto set_sub_partitions(Dim&& dim, size_t sub_partitions) noexcept -> void {
    blocks_per_part_[dim] = sub_partitions;
    blocks_per_part_set_  = true;
  }

  /**
   * Resizes the dimension to contain the given number of elements in the
   * dimension.
   *
   * \note This excludes the padding.
   *
   * \param  dim  The dimension to resize.
   * \param  size The size for the dimension (number of elements).
   * \tparam Dim  The type of the dimension specifier.
   */
  template <typename Dim>
  auto resize_dim(Dim&& dim, size_t size) noexcept -> void {
    space_.resize_dim(static_cast<Dim&&>(dim), size);
  }

  /**
   * Sets the padding for the tensor to the given amount.
   *
   * \note This amount is the amount of padding per side of each dimension, and
   *       is applied to each side of each dimension.
   *
   * \param padding The amount of padding for the tensor.
   */
  auto set_padding(size_t padding) noexcept -> void {
    space_.padding() = padding;
  }

  /**
   * Reallocates (or allocates if unallocated) the data for the tensor.
   * \param stream_map A map of gpu indicies to stream inidices which specifies
   *                   which streams to use for which gpu.
   */
  auto reallocate(const ripple::StreamMap& stream_map = ripple::StreamMap())
    -> void {
    initialize(stream_map);
  }

  /**
   * Computes the number of partitions for the tensor.
   *
   * \note The tensor will maps partitions to gpus, so this can also be used to
   *       determine the number of gpus.
   *
   * \return The total number of partitions for the tensor.
   */
  auto partitions() const noexcept -> Size {
    constexpr Size init = 1;
    return std::accumulate(
      partitions_.cbegin(), partitions_.cend(), init, std::multiplies<Size>());
  }

  /**
   * Computes the total number of elements per partition.
   * \return The total number of elements per partition.
   */
  auto partition_size() const -> Size {
    return std::accumulate(
      partition_sizes_.cbegin(),
      partition_sizes_.cend(),
      Size{1},
      std::multiplies<Size>());
  }

  /**
   * Computes the number of blocks in the given dimension.
   * \param  dim The dimension to get the number of blocks for.
   * \tparam Dim The type of the dimension specifier.
   * \return The number of blocks in the dimension.
   */
  template <typename Dim>
  auto blocks_in_dim(Dim&& dim) const noexcept -> size_t {
    return blocks_.size(static_cast<Dim&&>(dim));
  }

  /**
   * Gets an iterator to the blocks for the tensor.
   * \return A multi-dimensional iterator over the blocks for the tensor.
   */
  auto begin() noexcept -> BlockIter {
    return blocks_.begin();
  }

  /**
   * Computes the number of blocks *diagonally* across the tensor, i.e, the
   * hypotenuse of the computational space which the tensor defines.
   * \return The number of blocks diagonally for the tensor.
   */
  auto diagonal_blocks() const noexcept -> size_t {
    float sum = 0.0;
    unrolled_for<dims>(
      [&](auto dim) { sum += blocks_in_dim(dim) * blocks_in_dim(dim); });
    return static_cast<size_t>(std::ceil(std::sqrt(sum)));
  }

  /**
   * Computes the number of elements per block diagonally.
   * \return The number of elements in each block of the tensor, diagonally.
   */
  auto diagonal_block_size() const noexcept -> size_t {
    float sum = 0.0;
    unrolled_for<dims>(
      [&](auto dim) { sum += block_sizes_[dim] * block_sizes_[dim]; });
    return static_cast<size_t>(std::ceil(std::sqrt(sum)));
  }

  /**
   * Computes the number of cells in the tensor, diagonally.
   *
   * \note This *excludes* padding data.
   *
   * \return The number of elements in the tensor, diagonally.
   */
  auto diagonal_size() const noexcept -> size_t {
    float sum = 0.0;
    unrolled_for<dims>([&](auto dim) {
      sum += space_.internal_size(dim) * space_.internal_size(dim);
    });
    return static_cast<size_t>(std::ceil(std::sqrt(sum)));
  }

 private:
  Space      space_;                       //!< Space for the tensor.
  Partitions partitions_;                  //!< Partitions for the tensor.
  BlockSplit blocks_per_part_;             //!< Blocks per partition.
  Elements   partition_sizes_ = {};        //!< Sizes of partitions.
  Elements   block_sizes_     = {};        //!< Sizes of blocks.
  Blocks     blocks_;                      //!< Container for blocks.
  bool       blocks_per_part_set_ = false; //!< If blocks per part is set.

  /**
   * Initializes the tensor. This will allocate the blocks for the tensor, as
   * well as the data for each of the allocted blocks.
   * \param stream_map Map of stream indices for each gpu.
   */
  auto initialize(ripple::StreamMap stream_map = ripple::StreamMap()) -> void {
    blocks_.set_op_kind(BlockOpKind::synchronous);
    check_partitions();

    // Resize the block container:
    unrolled_for<dims>([&](auto dim) {
      if (!blocks_per_part_set_) {
        blocks_per_part_[dim] = default_blocks_per_partition;
      }

      block_sizes_[dim] =
        math::div_then_ceil(partition_sizes_[dim], blocks_per_part_[dim]);
      blocks_.resize_dim(
        dim, math::div_then_ceil(space_.internal_size(dim), block_sizes_[dim]));

      // assert(
      //  blocks_.size(dim) <= blocks_per_part_[dim] * partition_sizes_[dim] &&
      //  "Inavlid number of blocks per gpu!");
    });
    blocks_.reallocate_and_init();
    allocate_data_for_blocks(stream_map);
  }

  /**
   * Checks that the partitions are valid. If the partitions are invalid, then
   * the partitions are set to the default partitions, which is valid.
   *
   * \todo Add logging here to note invalid partition config.
   */
  auto check_partitions() -> void {
    const auto parts = partitions();
    if (parts > topology().num_gpus()) {
      assert(false && "More partitions specified than available gpus!");
    }

    if (parts <= topology().num_gpus()) {
      unrolled_for<dims>([&](auto dim) {
        partition_sizes_[dim] =
          math::div_then_ceil(space_.internal_size(dim), partitions_[dim]);
      });
      return;
    }

    set_default_partition();
  };

  /**
   * Computes the dimension with the partitions with the largest number of
   * elements.
   * \return The dimension with the partitions with the largest number of
   *         elements.
   */
  auto largest_partition_dim() const noexcept -> size_t {
    size_t index = 0;
    auto   max   = partition_sizes_[0];
    unrolled_for<dims - 1>([&](auto d) {
      constexpr size_t dim = d + 1;
      if (partition_sizes_[dim] > max) {
        max   = partition_sizes_[dim];
        index = dim;
      }
    });
    return index;
  }

  /**
   * Sets the partitioning scheme to the default, which is to split the largest
   * dimension until the number of elements is less than the min split
   * threshold.
   */
  auto set_default_partition() noexcept -> void {
    for (size_t i = 0; i < dims; ++i) {
      partitions_[i]      = 1;
      partition_sizes_[i] = space_.internal_size(i);
    }

    // Find the scaling factor:
    size_t scaling_factor = 2;
    while (topology().num_gpus() % scaling_factor != 0) {
      scaling_factor++;
    }

    size_t dim = 0;
    while (partition_size() > min_split_threshold) {
      // Split the latgest dimension, this assumes that the number of gpus are
      // a power of two
      dim = largest_partition_dim();
      partitions_[dim] *= scaling_factor;

      if (partitions() > topology().num_gpus()) {
        partitions_[dim] /= scaling_factor;
        return;
      }

      partition_sizes_[dim] /= scaling_factor;
    }
  }

  /**
   * Allocates the data for the blocks and initializes the fields for the
   * blocks.
   * \param stream_map A map of gpu ids to streams to use for the partitions
   *                   of the tensor.
   */
  auto allocate_data_for_blocks(const ripple::StreamMap& stream_map) -> void {
    invoke(blocks_, [&](auto block) {
      auto& host   = block->host_data;
      auto& device = block->device_data;

      // Set the host component of the block to enable asynchronous operations
      // so that compute and transfer can be overlapped:
      host.set_op_kind(BlockOpKind::asynchronous);

      // Now set the padding for the block:
      host.set_padding(space_.padding());
      device.set_padding(space_.padding());

      size_t prev_dim_partitions = 1, id = 0;
      unrolled_for<dims>([&](auto dim) {
        block->indices[dim]      = global_idx(dim);
        block->block_sizes[dim]  = block_sizes_[dim];
        block->global_sizes[dim] = space_.internal_size(dim);
        block->max_indices[dim]  = blocks_.size(dim) - 1;

        const Size elements_start = global_idx(dim) * block_sizes_[dim];
        const Size block_size     = std::min(
          space_.internal_size(dim) - elements_start, block_sizes_[dim]);

        host.resize_dim(dim, block_size);
        device.resize_dim(dim, block_size);

        id += block->indices[dim] / blocks_per_part_[dim] * prev_dim_partitions;
        prev_dim_partitions *= partitions_[dim];
      });

      // Set all the gpu data:
      GpuInfo& gpu = topology().gpus[id];
      block->set_device_id(gpu.index);
      block->set_transfer_stream(
        gpu.transfer_streams[gpu.next_transfer_stream_id()].stream);
      gpu::set_device(gpu.index);

      // Allocate the host memory:
      host.reallocate();

      // Here we either use the supplied stream, or the first one.
      auto stream_id =
        stream_map.find(id) != stream_map.end() ? stream_map.at(id) : 0;

      // Now alloate device data:
      auto& stream = gpu.compute_streams[stream_id].stream;
      device.set_stream(stream);
      device.reallocate();
      gpu::synchronize_stream(stream);

      block->data_state = DataState::updated_device;
    });
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_TENSOR_HPP
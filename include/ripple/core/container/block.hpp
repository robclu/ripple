//==--- ripple/core/container/block.hpp -------------------- -*- C++ -*- ---==//
//
//                                  Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block.hpp
/// \brief This file defines a class for a block which has both host and device
///        data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_BLOCK_HPP
#define RIPPLE_CONTAINER_BLOCK_HPP

#include "device_block.hpp"
#include "host_block.hpp"
#include "memcopy_padding.hpp"
#include <ripple/core/algorithm/reduce.hpp>
#include <ripple/core/arch/topology.hpp>
#include <ripple/core/iterator/indexed_iterator.hpp>

namespace ripple {

/*==--- [Multiblock traits specialization] ---------------------------------==*/

/*
 * Specialization of the block enabled traits for a block.
 * \param  T          The type of the data in the block.
 * \tparam Dimensions The number of dimensions in the block.
 */
template <typename T, size_t Dimensions>
struct MultiBlockTraits<Block<T, Dimensions>> {
 private:
  // clang-format off
  /** Defines the type of the host block for the block. */
  using LayoutTraits = layout_traits_t<T>;
  /** Defines the value type of the block data. */
  using IterValue    = typename LayoutTraits::Value;
  /** Defines the type of the space used by the block. */
  using Space        = DynamicMultidimSpace<Dimensions>;

 public:
  /** The number of dimensions in the block. */
  static constexpr size_t dimensions = Dimensions;

  // clang-format off
  /** Defines the value type of the block data. */
  using Value          = T;
  /** Defines the type of the iterator over global block data. */
  using Iterator       = IndexedIterator<IterValue, Space>;
  /** Defines the type of  the iterator over shared block data. */
  using SharedIterator = BlockIterator<IterValue, Space>;
};

/**
 * Defines the state of the data for a parition.
 */
enum class DataState : uint8_t {
  invalid          = 0, //!< Data is not valid.
  updated_host     = 1, //!< Data is on the host and is updated.
  submitted_host   = 2, //!< Data has been submitted on the host.
  updated_device   = 3, //!< Data is on the device and is updated.
  submitted_device = 4  //!< Data has been submitted on the device.
};

/**
 * Defines a type which wraps a host and device block into a single type, and
 * which has a state for the different regions of data in the parition.
 *
 * This class is designed to partition both the computational space as well
 * as the memory spaces, and to be used as a building block for the Tensor
 * class.
 *
 * \tparam T          The type of the data for the block.
 * \tparam Dimensions The number of dimensions for the block.
 */
template <typename T, size_t Dimensions>
struct Block : MultiBlock<Block<T, Dimensions>> {
  /** Defines the traits for the block. */
  using Traits = MultiBlockTraits<Block<T, Dimensions>>;
 public:
  // clang-format off
  /** Defines the type of the type for the block index. */
  using Index          = std::array<uint32_t, Dimensions>;
  /** Defines the type of the host block for the block. */
  using HostBlock      = HostBlock<T, Dimensions>;
  /** Defines the type of the device block for the block. */
  using DeviceBlock    = DeviceBlock<T, Dimensions>;
  /** Defines the type of the host iterator. */
  using Iterator       = typename Traits::Iterator;
  /** Defines the type of the iterator over shared memory. */
  using SharedIterator = typename Traits::SharedIterator;
  /** Defines the type of the stream. */
  using Stream         = typename DeviceBlock::Stream;
  // clang-format on

  /** Defines the number of dimension for the block. */
  static constexpr size_t dims = Dimensions;

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor for the block.
   */
  Block() = default;

  /**
   * Destructor -- defaulted.
   */
  ~Block() = default;

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Ensures that the data is available on the device.
   *
   * \note This will make a copy from the host to the device if the data is not
   *       already on the device.
   */
  auto ensure_device_data_available() noexcept -> void {
    if (data_state == DataState::updated_host) {
      device_data.copy_data(host_data);
      data_state = DataState::updated_device;
    }
  }

  /**
   * Ensures that the data is available on the host.
   *
   * \note This will make a copy from the device to the host if the data is not
   *       already on the host.
   */
  auto ensure_host_data_available() noexcept -> void {
    if (data_state == DataState::updated_device) {
      host_data.copy_data(device_data);
      data_state = DataState::updated_host;
    }
  }

  /**
   * Determines if the block has padding.
   * \return true if the block has padding, otherwise false.
   */
  auto has_padding() const noexcept -> bool {
    return host_data.padding() > 0;
  }

  /**
   * Sets the padding for the block to be the given amount.
   *
   * \note The amount here is the number of elements of a single side of a
   *       single dimension, which will be applied to all sides of all
   *       dimensions.
   *
   * \param amount The amount of  padding.
   */
  auto set_padding(size_t amount) noexcept -> void {
    host_data.set_padding(amount);
    device_data.set_padding(amount);
  }

  /**
   * Gets the amount of padding for a single side of a single dimension for the
   * block.
   * \return The amount of padding for the block.
   */
  auto padding() const noexcept -> size_t {
    return device_data.padding();
  }

  /**
   * Sets the device id for the block.
   * \param device_id The id of the device for the block.
   */
  auto set_device_id(uint32_t device_id) noexcept -> void {
    gpu_id = device_id;
    device_data.set_device_id(device_id);
  }

  /**
   * Gets an iterator to the device data for the block.
   * \return An iterator to the first valid (non-padding) element in the block.
   */
  auto device_iterator() const noexcept -> Iterator {
    auto iter = Iterator{device_data.begin()};
    set_iter_properties(iter);
    return iter;
  }

  /**
   * Gets an iterator to the host data.
   * \return An iterator to the first valid (non-padding) element in the block.
   */
  auto host_iterator() const noexcept -> Iterator {
    auto iter = Iterator{host_data.begin()};
    set_iter_properties(iter);
    return iter;
  }

  /**
   * Gets an iterator to the beginning of the device data.
   * \return An iterator to the beginning of the device data.
   */
  // decltype(auto) begin() noexcept {
  //  return device_data.begin();
  //}

  /**
   * Returns the stream for this block.
   * \return A reference to the stream for the block.
   */
  auto stream() const noexcept -> Stream {
    return device_data.stream();
  }

  /**
   * Synchronizes the block's stream.
   *
   * \todo Change this to sync both compute and copy streams.
   */
  auto synchronize() noexcept -> void {
    gpu::set_device(gpu_id);
    gpu::synchronize_stream(stream());
  }

  /**
   * Returns the total number of elements in the given dimension.
   * \param dim The dimension to get the size of.
   * \param Dim The type of the dimension specifier.
   * \return The number of elements in teh given dimenion, *excluding* padding
   *         for the dimsnion
   */
  template <typename Dim>
  auto size(Dim&& dim) const noexcept -> size_t {
    return device_data.size(ripple_forward(dim));
  }

  /**
   *  Determines if the block is the last block in the given dimension.
   * \param  dim The dimension to check if this is the last block in.
   * \tparam Dim The type of the dimension specifier.
   * \return true if this block is the last block in the dimension.
   */
  template <typename Dim>
  auto last_in_dim(Dim&& dim) const noexcept -> bool {
    return indices[dim] == max_indices[dim];
  }

  /**
   * Returns true if the block is the first block in the given dimension.
   * \param  dim The dimension to check if this is the last block in.
   * \tparam Dim The type of the dimension specifier.
   * \return true if this block is the first block in the dimension.
   */
  template <typename Dim>
  auto first_in_dim(Dim&& dim) const noexcept -> bool {
    return indices[dim] == 0;
  }

  /*==--- [padding copying] ------------------------------------------------==*/

  /**
   * Fills the padding for this block using the other block, for the specified
   * face.
   *
   * \note Whether this copies the data from the inside of the domain or from
   *       the padding of the other block depending on the mapping.
   *
   * \param  other     The other block to use to fill the padding.
   * \param  dest_face The destination face to fill.
   * \tparam Dim       The dimension of the destination face.
   * \tparam Location  The location of the face to fill the padding for.
   * \tparam Map       The padding for the face.
   */
  template <size_t Dim, Face Location, Mapping Map>
  auto fill_padding(
    Block&                            other,
    FaceSpecifier<Dim, Location, Map> dest_face,
    ExecutionKind                     exe) noexcept -> void {
    if (exe == ExecutionKind::gpu) {
      fill_padding_device(other, dest_face);
      return;
    }
    fill_padding_host(other, dest_face);
  }

  /*==--- [reduction] ------------------------------------------------------==*/

  /**
   * Performs a reduction on either the host or the device, depending on the
   * kind of the exection.
   * \param  exe  The kind of the execution.
   * \param  pred The predicate for the reduction.
   * \param  as   Arguments for the predicate.
   * \tparam Pred The type of the predicate.
   * \tparam As   The type of the predicate arguments.
   ** \return The results of the reduction of the data.
   */
  template <typename Pred, typename... As>
  auto reduce(ExecutionKind exec, Pred&& pred, As&&... as) noexcept -> T {
    // clang-format off
    return exec == ExecutionKind::gpu
      ? reduce_on_device(ripple_forward(pred), ripple_forward(as)...)
      : reduce_on_host(ripple_forward(pred), ripple_forward(as)...);
    // clang-format on
  }

  /**
   * Performs a reduction of the block device data, returning the result.
   *
   * \note If the data is currently on the host then this will first copy the
   *       data to the device.
   *
   * \param  pred The predicate for the reduction.
   * \param  args Arguments for the predicate.
   * \tparam Pred The type of the predicate.
   * \tparam Args The type of the predicate arguments.
   * \return The result of the reduction of the data.
   */
  template <typename Pred, typename... Args>
  auto reduce_on_device(Pred&& pred, Args&&... args) noexcept -> T {
    ensure_device_data_available();
    return ::ripple::reduce(
      device_data, ripple_forward(pred), ripple_forward(args)...);
  }

  /**
   * Performs a reduction of the host device data, returning the result.
   *
   * \note If the data is curently on the device then this will first copy the
   *       data to the host.
   *
   * \param  pred The predicate for the reduction.
   * \param  args Arguments for the predicate.
   * \tparam Pred The type of the predicate.
   * \tparam Args The type of the predicate arguments.
   * \return The result of the reduction of the data.
   */
  template <typename Pred, typename... Args>
  auto reduce_on_host(Pred&& pred, Args&&... args) noexcept -> T {
    ensure_host_data_available();
    return ::ripple::reduce(
      host_data, ripple_forward(pred), ripple_forward(args)...);
  }

  //==--- [members] --------------------------------------------------------==//

  HostBlock   host_data;         //!< Host block data.
  DeviceBlock device_data;       //!< Device block data.
  Index       indices      = {}; //!< Indices of the block.
  Index       block_sizes  = {}; //!< Sizes of the blocks.
  Index       global_sizes = {}; //!< Global sizes.
  Index       max_indices  = {}; //!< Max indices for each dimension.
  uint32_t    gpu_id       = 0;  //!< Device index.
  DataState   data_state   = DataState::invalid; //!< Data state.

 private:
  /**
   * Creates a face specifier using the given Location, with a face
   * *opposite* to the one given in the face location.
   *
   * For example if the location is Face::start, the returned specifier has
   * Face::end ince the two specifiers must be neighbours.
   *
   * \tparam Dim      The dimension to get the face for.
   * \tparam Location The FaceLocation used to creat the specifier.
   * \tparam Map      The mapping for the face.
   * \return The opposite face specifier.
   */
  template <size_t Dim, Face Location, Mapping Map>
  static constexpr auto
  opp_face_for_src(FaceSpecifier<Dim, Location, Map>) noexcept {
    constexpr auto location = Location == Face::start ? Face::end : Face::start;
    return FaceSpecifier<Dim, location, Mapping::domain>{};
  }

  /**
   * Creates a face specifier using the given Location, with a face
   * *the same as* the one given in the face location.
   *
   * For example if the location is Face::start, the returned specifier also has
   * a location Face::start.
   *
   * \tparam Dim      The dimension to get the face for.
   * \tparam Location The FaceLocation used to creat the specifier.
   * \tparam Map      The mapping for the face.
   * \return The same face specifier.
   */
  template <size_t Dim, Face Location, Mapping Map>
  static constexpr auto
  same_face_for_dst(FaceSpecifier<Dim, Location, Map>) noexcept {
    return FaceSpecifier<Dim, Location, Mapping::padding>{};
  }

  /**
   * Sets the properties for the given iterator.
   * \param  it       The iterator to set the properties for.
   * \tparam Iterator The type of the iterator.
   */
  template <typename Iterator>
  auto set_iter_properties(Iterator& it) const noexcept -> void {
    unrolled_for<dims>([&](auto dim) {
      it.set_block_start_index(dim, indices[dim] * block_sizes[dim]);
      it.set_global_size(dim, global_sizes[dim]);
    });
  }

  /**
   * Fills the padding for this block using the other block, for the specified
   * face.
   *
   * \note Note, this implementation is for execution on the device.
   *
   * \param  other     The other block to use to fill the padding.
   * \param  dest_face The destination face to fill.
   * \tparam Dim       The dimension of the destination face.
   * \tparam Location  The location of the face to fill the padding for.
   * \tparam Map       The padding for the face.
   */
  template <size_t Dim, Face Location, Mapping Map>
  auto fill_padding_device(
    Block& other, FaceSpecifier<Dim, Location, Map> dest_face) noexcept
    -> void {
    constexpr auto src_face = opp_face_for_src(dest_face);
    constexpr auto dst_face = same_face_for_dst(dest_face);

    // If we are on the same gpu, then we can do the device to device copy,
    // otherwise we need to go through the host:
    if (topology().device_to_device_available(gpu_id, other.gpu_id)) {
      gpu::set_device(other.gpu_id);
      memcopy_padding(
        other.device_data,
        device_data,
        src_face,
        dst_face,
        other.device_data.stream());
      if (other.gpu_id == gpu_id) {
        gpu::synchronize_stream(other.device_data.stream());
      }
      return;
    }

    // Here we can't do a device -> device copy, so go through the host:
    // First copy from the other block's device data to this block's host data:
    gpu::set_device(other.gpu_id);
    memcopy_padding(
      other.device_data,
      host_data,
      src_face,
      src_face,
      other.device_data.stream());

    // Have to wait for the copy to finish ...
    gpu::synchronize_stream(other.device_data.stream());

    // Then copy from this block's host data to this blocks device data:
    gpu::set_device(gpu_id);
    memcopy_padding(
      host_data, device_data, src_face, dst_face, device_data.stream());
  }

  /**
   * Fills the padding for this block using the other block, for the specified
   * face.
   *
   * \note Note, this implementation is for execution on the host
   *
   * \param  other     The other block to use to fill the padding.
   * \param  dest_face The destination face to fill.
   * \tparam Dim       The dimension of the destination face.
   * \tparam Location  The location of the face to fill the padding for.
   * \tparam Map       The padding for the face.
   */
  template <size_t Dim, Face Location, Mapping Map>
  auto fill_padding_host(
    Block& other, FaceSpecifier<Dim, Location, Map> dest_face) noexcept
    -> void {
    constexpr auto src_face = opp_face_for_src(dest_face);
    constexpr auto dst_face = same_face_for_dst(dest_face);

    // Here we know that we have to go through the host.
    // First make sure that the host data is up to date. We start both copies
    // and then wait for both copies:
    other.ensure_host_data_available();
    ensure_device_data_available();
    other.synchronize();
    synchronize();

    // Then copy from this block's host data to this blocks device data:
    memcopy_padding(host_data, device_data, src_face, dst_face);
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BLOCK_HPP

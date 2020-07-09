//==--- ripple/core/container/block.hpp -------------------- -*- C++ -*- ---==//
//
//                                  Ripple
//
//                      Copyright (c) 2020 Rob Clucas
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

#include "block_traits.hpp"
#include "device_block.hpp"
#include "host_block.hpp"
#include <ripple/core/arch/topology.hpp>
#include <ripple/core/iterator/indexed_iterator.hpp>

namespace ripple {

/// Defines the state of the block data.
enum class BlockState : uint8_t {
  invalid          = 0, //!< Data is not valid.
  updated_host     = 1, //!< Data is on the host and is updated.
  submitted_host   = 2, //!< Data has been submitted on the host.
  updated_device   = 3, //!< Data is on the device and is updated.
  submitted_device = 4  //!< Data has been submitted on the device.
};

/// This struct is a buffer which is used to copy data between the

/// Defines a type which wraps a host and device block into a single type, and
/// which has a state for the different data components in the block.
///
/// The Block type stores both host and device block data, and provides
/// operations to move the data between the host and device memory spaces.
///
/// \tparam T     The type of the data for the block.
/// \tparam Dims  The number of dimensions for the block.
template <typename T, size_t Dimensions>
struct Block : public BlockEnabled<Block<T, Dimensions>> {
  // clang-format off
  /// Defines the type of the type for the block index.
  using index_t        = std::array<uint32_t, Dimensions>;
  /// Defines the type of the host block for the grid.
  using host_block_t   = HostBlock<T, Dimensions>;
  /// Defines the type of the device block for the grid.
  using device_block_t = DeviceBlock<T, Dimensions>;
  /// Defines the type of the space used by the host block iterator.
  using host_space_t   = typename block_traits_t<host_block_t>::space_t;
  /// Defines the type of the space used by the device block iterator.
  using device_space_t = typename block_traits_t<device_block_t>::space_t;
  /// Defines the type of the host iterator.
  using host_iter_t    = IndexedIterator<T, host_space_t>;
  /// Defines the type of the device iterator.
  using device_iter_t  = IndexedIterator<T, device_space_t>;
  /// Defines the type of the stream.
  using stream_t       = typename device_block_t::stream_t;
  // clang-format on

  /// Defines the number of dimension for the block.
  static constexpr size_t dims = Dimensions;

  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor for a block.
  Block() = default;

  /// Default destructor, sets the device to the block device for deallocation.
  ~Block() {
    cudaSetDevice(gpu_id);
  }

  //==--- [interface] ------------------------------------------------------==//

  /// Ensures that the data is available on the device. This will make a copy
  /// from the host to the device if the data is not already on the device.
  auto ensure_device_data_available() -> void {
    if (data_state == BlockState::updated_host) {
      cudaSetDevice(gpu_id);
      device_data.copy_data(host_data);
    }
  }

  /// Ensures that the data is available on the host. This will make a copy
  /// from the device to the host if the data is not already on the host.
  auto ensure_host_data_available() -> void {
    if (data_state == BlockState::updated_device) {
      cudaSetDevice(gpu_id);
      host_data.copy_data(device_data);
    }
  }

  /// Returns true if the block has padding, otherwise returns false.
  auto has_padding() const -> bool {
    return host_data.padding() > 0;
  }

  /// Sets the padding for the block to be \p width elements.
  /// \param width The width of the padding.
  auto set_padding(size_t width) -> void {
    host_data.set_padding(width);
    device_data.set_padding(width);
  }

  /// Returns the amount of padding for the block.
  auto padding() const -> size_t {
    return device_data.padding();
  }

  /// Returns an iterator to the device data for the block.
  auto device_iterator() const -> device_iter_t {
    auto iter = device_iter_t{device_data.begin()};
    set_iter_properties(iter);
    return iter;
  }

  /// Returns an iterator to the host data.
  auto host_iterator() const -> host_iter_t {
    auto iter = host_iter_t{host_data.begin()};
    set_iter_properties(iter);
    return iter;
  }

  /// Returns an iterator to
  auto begin() {
    return device_data.begin();
  }

  /// Returns the stream for this block.
  auto stream() const {
    return device_data.stream();
  }

  //==--- [size & indices] -------------------------------------------------==//

  /// Returns the total number of elements in dimension \p dim.
  /// \param dim The dimension to get the size of.
  /// \param Dim The type of the dimension specifier.
  template <typename Dim>
  auto size(Dim&& dim) const -> std::size_t {
    return device_data.size(std::forward<Dim>(dim));
  }

  /// Returns true if the block is the last block in \p dim dimension.
  /// \param  dim The dimension to check if this is the last block in.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  auto last_in_dim(Dim&& dim) const noexcept -> bool {
    return indices[dim] == max_indices[dim];
  }

  /// Returns true if the block is the first block in \p dim dimension.
  /// \param  dim The dimension to check if this is the last block in.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  auto first_in_dim(Dim&& dim) const noexcept -> bool {
    return indices[dim] == 0;
  }

  //==--- [padding copying] ------------------------------------------------==//

  /// Fills the padding for this block using the \p other block, for the face
  /// spcified by \p dst_face.
  /// \param other    The other block to use to fill the padding.
  /// \param dst_face The destination face to fill.
  /// \tparam Dim     The dimension of the destination face.
  /// \tparam Face    The face in the dimension.
  template <size_t Dim, FaceLocation Face>
  auto
  fill_padding(const Block& other, FaceSpecifier<Dim, Face> dst_face) -> void {
    constexpr auto src_face = opposite_face(dst_face);

    // If we are on the same gpu, then we can do the device to device copy,
    // otherwise we need to go through the host:
    if (
      gpu_id == other.gpu_id ||
      topology().gpus[gpu_id].peer_to_peer_available(other.gpu_id)) {
      cudaSetDevice(other.gpu_id);
      other.device_data.copy_padding_into_device(
        device_data, src_face, dst_face);
    } else {
      cudaSetDevice(other.gpu_id);
      // First copy from the other block's device to the other block's host:
      other.device_data.copy_padding_into_host(host_data, src_face, src_face);
      cudaStreamSynchronize(other.device_data.stream());

      cudaSetDevice(gpu_id);
      // Then copy from the other block's host to this blocks device:
      host_data.copy_padding_into_device(device_data, src_face, dst_face);
    }
  }

  //==--- [members] --------------------------------------------------------==//

  host_block_t   host_data;            //!< Host block data.
  device_block_t device_data;          //!< Device block data.
  index_t        indices;              //!< Indices of the block.
  index_t        block_sizes;          //!< Sizes of the blocks.
  index_t        global_sizes;         //!< Global sizes.
  index_t        max_indices;          //!< Max indices for each dimension.
  Block*         sibling    = nullptr; //!< Data sharer.
  uint32_t       gpu_id     = 0;       //!< Device index.
  BlockState     data_state = BlockState::invalid; //!< Data state.

 private:
  /// Creates a face specifier using the given FaceLocation, with a face
  /// __opposite__ to the one given in the face location. I.e, if the
  /// FaceLocation is FaceLocation::start, the returned specifier has
  /// FaceLocation::end, since the two specifiers must be neighbours.
  /// \tparam Face The FaceLocation used to creat the specifier.
  template <size_t Dim, FaceLocation Face>
  static constexpr auto opposite_face(FaceSpecifier<Dim, Face>) {
    constexpr auto location = Face == FaceLocation::start ? FaceLocation::end
                                                          : FaceLocation::start;
    return FaceSpecifier<Dim, location>();
  }

  /// Sets the properties for the \p it iterator.
  /// \param  it       The iterator to set the properties for.
  /// \tparam Iterator The type of the iterator.
  template <typename Iterator>
  auto set_iter_properties(Iterator& it) const -> void {
    unrolled_for<dims>([&](auto dim) {
      it.set_block_start_index(dim, indices[dim]);
      it.set_global_size(dim, global_sizes[dim]);
    });
  }
};

//==--- [Block enabled trairs specialization] ------------------------------==//

/// Specialization of the block enabled traits for a block.
/// \param  T    The type of the data in the block.
/// \tparam Dims The number of dimensions in the block.
template <typename T, size_t Dims>
struct BlockEnabledTraits<Block<T, Dims>> {
  /// The number of dimensions in the block.
  static constexpr size_t dimensions = Dims;

  /// Defines the value type of the block data.
  using value_t = T;
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BLOCK_HPP

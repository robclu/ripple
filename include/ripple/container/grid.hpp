//==--- ripple/container/grid.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                  Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  grid.hpp
/// \brief This file defines a class which represents a grid.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_GRID_HPP
#define RIPPLE_CONTAINER_GRID_HPP

#include "grid_traits.hpp"
#include "device_block.hpp"
#include "host_block.hpp"


namespace ripple {

/// Implementation of the grid class.
/// \tparam T          The type of the data for the grid.
/// \tparam Dimensions The number of dimensions for the grid.
template <typename T, size_t Dimensions>
class Grid {
  /// Defines the type of the host block for the grid.
  using host_block_t   = HostBlock<T, Dimensions>;
  /// Defines the type of the device block for the grid.
  using device_block_t = DeviceBlock<T, Dimensions>;

  /// Defines a type which wraps a host and device block into a single type, and
  /// which has a state for the different data components in the block.
  struct Block {
    /// Defines the state of the block data.
    enum class State : uint8_t {
      updated_host       = 0,   //!< Data is on the host and is updated.
      not_updated_host   = 1,   //!< Data is on the host but is not updated.
      updated_device     = 2,   //!< Data is on the device and is updated.
      not_updated_device = 3    //!< Data is on the device but is not updated.
    };

    host_block_t   host_data;   //!< Host block data.
    device_block_t device_data; //!< Device block data.

    State data_state    = State::not_updated_host; //!< Data state.
    State padding_state = State::not_updated_host; //!< Padding state.
  };

  /// Defines the type of the container used for the blocks.
  using blocks_t = HostBlock<Block, Dimensions>;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Initializes the size of each of the dimensions of the grid, allocating the
  /// memory for the grid. This constructor is only enabled when the number of
  /// arguments matches the dimensionality of the grid and the \p sizes are
  /// numeric types.
  ///
  /// \param  sizes The sizes of the dimensions for the grid.
  /// \tparam Sizes The types of other dimension sizes.
  template <
    typename... Sizes,      
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  Grid(Sizes&&... sizes) 
  : _data{BlockOpKind::asynchronous, std::forward<Sizes>(sizes)...} {}

  /// Initializes the size of each of the dimensions of the grid, allocating the
  /// memory for the grid, with padding data for the boundary of the grid. This
  /// constructor is only enabled when the number of arguments matches the
  /// dimensionality of the grid and the \p sizes are numeric types.
  ///
  /// \param  padding The amount of padding for the grid.
  /// \param  sizes   The sizes of the dimensions for the grid.
  /// \tparam Sizes   The types of other dimension sizes.
  template <
    typename... Sizes,      
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  Grid(size_t padding, Sizes&&... sizes) 
  : _data{BlockOpKind::asynchronous, padding, std::forward<Sizes>(sizes)...} {}

  //==--- [interface] ------------------------------------------------------==//
  
   /// Returns the total number of elements in the grid.
  auto size() const -> std::size_t {
    return _data.size();
  } 

  /// Returns the total number of elements in dimension \p dim.
  /// \param dim The dimension to get the size of.
  /// \param Dim The type of the dimension specifier.
  template <typename Dim>
  auto size(Dim&& dim) const -> std::size_t {
    return _data.size(std::forward<Dim>(dim));
  }
 
 private:
  host_block_t _data;     //!< Host side data for the block.
  blocks_t     _blocks;   //!< Blocks for the grid.
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_GRID_HPP


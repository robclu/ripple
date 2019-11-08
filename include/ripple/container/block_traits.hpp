//==--- ripple/container/block_traits.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block_traits.hpp
/// \brief This file defines traits and forward declarations for blocks.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_BLOCK_TRAITS_HPP
#define FLUIDITY_CONTAINER_BLOCK_TRAITS_HPP

#include <ripple/multidim/dynamic_multidim_space.hpp>
#include <ripple/storage/storage_traits.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// Definition of a device block class which stores multidimensional data on
/// the host. This will store the data in a strided format if the type T
/// implements the StridableLayout interface and the descriptor for the storage
/// for the type has a StorageLayout::strided_view type, otherwise this will
/// store the data in a contiguous format.
///
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions> class DeviceBlock;  

/// Definition of a host block class which stores multidimensional data on
/// the host. This will store the data in a strided format if the type T
/// implements the StridableLayout interface and the descriptor for the storage
/// for the type has a StorageLayout::strided_view type, otherwise this will
/// store the data in a contiguous format.
///
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions> class HostBlock;

/// The BlockTraits class defines traits for a block.
///
/// \tparm Block The block to get the traits for.
template <typename Block> struct BlockTraits;

//==--- [specializations] --------------------------------------------------==//

/// Specialization of the BlockTraits struct for a host block.
/// \tparam T The data type for the block.
/// \tparam Dimensions The number of dimensions for the block.
template <typename T, std::size_t Dimensions>
struct BlockTraits<HostBlock<T, Dimensions>> {
 private:
  /// Defines the allocation traits for the type.
  using layout_traits_t = layout_traits_t<T>;
 public:
  /// Defines the value type of the block.
  using value_t     = typename layout_traits_t::value_t;
  /// Defines the type of allocator for the tensor.
  using allocator_t = typename layout_traits_t::allocator_t;
  /// Defines the the of the dimension information for the tensor.
  using space_t     = DynamicMultidimSpace<Dimensions>;
};

/// Specialization of the BlockTraits struct for a device block.
/// \tparam T The data type for the block.
/// \tparam Dimensions The number of dimensions for the block.
template <typename T, std::size_t Dimensions>
struct BlockTraits<DeviceBlock<T, Dimensions>> {
 private:
  /// Defines the allocation traits for the type.
  using layout_traits_t = layout_traits_t<T>;
 public:
  /// Defines the value type of the block.
  using value_t     = typename layout_traits_t::value_t;
  /// Defines the type of allocator for the tensor.
  using allocator_t = typename layout_traits_t::allocator_t;
  /// Defines the the of the dimension information for the tensor.
  using space_t     = DynamicMultidimSpace<Dimensions>;
};

//==--- [aliases] ----------------------------------------------------------==//

/// Alias for a 1-dimensional host side block.
/// \tparam T The type of the data for the block.
template <typename T>
using host_block_1d_t = HostBlock<T, 1>;

/// Alias for a 1-dimensional device side block.
/// \tparam T The type of the data for the block.
template <typename T>
using device_block_1d_t = DeviceBlock<T, 1>;

/// Alias for a 2-dimensional host side block.
/// \tparam T The type of the data for the block.
template <typename T>
using host_block_2d_t = HostBlock<T, 2>;

/// Alias for a 2-dimensional device side block.
/// \tparam T The type of the data for the block.
template <typename T>
using device_block_2d_t = DeviceBlock<T, 2>;

/// Alias for a 3-dimensional host side block.
/// \tparam T The type of the data for the block.
template <typename T>
using host_block_3d_t = HostBlock<T, 3>;

/// Alias for a 3-dimensional device side block.
/// \tparam T The type of the data for the block.
template <typename T>
using device_block_3d_t = DeviceBlock<T, 3>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_BLOCK_TRAITS_HPP

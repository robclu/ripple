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

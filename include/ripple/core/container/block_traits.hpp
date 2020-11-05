//==--- ripple/core/container/block_traits.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block_traits.hpp
/// \brief This file defines traits and forward declarations for blocks.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_BLOCK_TRAITS_HPP
#define RIPPLE_CONTAINER_BLOCK_TRAITS_HPP

#include <ripple/core/iterator/block_iterator.hpp>
#include <ripple/core/multidim/dynamic_multidim_space.hpp>
#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/*==--- [forward declarations] ---------------------------------------------==*/

/**
 * Definition of a device block class which stores multidimensional data on
 * the device.
 *
 * This will store the data in a strided format if the type T
 * implements the StridableLayout interface and the descriptor for the storage
 * for the type has a StorageLayout::StridedView type, otherwise this will
 * store the data in a contiguous format.
 *
 * \tparam T          The type of the data stored in the tensor.
 * \tparam Dimensions The number of dimensions in the tensor.
 */
template <typename T, size_t Dimensions>
class DeviceBlock;

/**
 * Definition of a host block class which stores multidimensional data on
 * the host.
 *
 * This will store the data in a strided format if the type T
 * implements the StridableLayout interface and the descriptor for the storage
 * for the type has a StorageLayout::StridedView type, otherwise this will
 * store the data in a contiguous format.
 *
 * \tparam T          The type of the data stored in the tensor.
 * \tparam Dimensions The number of dimensions in the tensor.
 */
template <typename T, size_t Dimensions>
class HostBlock;

/*==--- [block traits] -----------------------------------------------------==*/

/**
 * Defines an enum for the type of operations for a block.
 */
enum class BlockOpKind : uint8_t {
  asynchronous = 0, //!< Asynchrnous operations for a block.
  synchronous  = 1  //!< Synchrnous operations for a block.
};

/**
 * The BlockTraits class defines traits for a block. This instance is for
 * non-specialzied blocks so that that can be used with any type.
 * \tparm T The type to get the block traits for.
 */
template <typename T>
struct BlockTraits {
  // clang-format off
  /** Defines the value type of the block. */
  using Value     = void*;
  /** Defines the type of allocator for the block. */
  using Allocator = void*;
  /** Defines the the of the dimension information for the block. */
  using Space     = void*;
  /** Defines the type of the iterator for the block. */
  using Iter      = BlockIterator<Value, Space>;

  /** Defines the number of dimensions for the block. */
  static constexpr auto dimensions      = 0;
  /** Defines the the type is *not* a block. */
  static constexpr auto is_block        = false;
  /** Defines that the blocks is *not* a host block. */
  static constexpr auto is_host_block   = false;
  /** Defines that the block is *not* a device block */
  static constexpr auto is_device_block = false;
  // clang-format on
};

/**
 * Specialization of the BlockTraits struct for a host block.
 * \tparam T          The data type for the block.
 * \tparam Dimensions The number of dimensions for the block.
 */
template <typename T, size_t Dimensions>
struct BlockTraits<HostBlock<T, Dimensions>> {
 private:
  /** Defines the layout traits for the type. */
  using LayoutTraits = layout_traits_t<T>;

 public:
  // clang-format off
  /** Defines the value type of the block. */
  using Value     = typename LayoutTraits::Value;
  /** Defines the type of allocator for the block. */
  using Allocator = typename LayoutTraits::Allocator;
  /** Defines the the of the space information for the block. */
  using Space     = DynamicMultidimSpace<Dimensions>;
  /** Defines the type of the iterator for the block. */
  using Iter      = BlockIterator<Value, Space>;

  /** Defines the number of dimensions for the block. */
  static constexpr size_t dimensions      = Dimensions;
  /** Defines that the traits are for a valid block. */
  static constexpr bool   is_block        = true;
  /** Defines that the blocks is a host block. */
  static constexpr bool   is_host_block   = true;
  /** Defines that the block is not a device block. */
  static constexpr bool   is_device_block = false;
  /** Defines the alignment for the block. */
  static constexpr size_t alignment       = LayoutTraits::alignment;
  // clang-format on
};

/**
 * Specialization of the BlockTraits struct for a device block.
 * \tparam T The data type for the block.
 * \tparam Dimensions The number of dimensions for the block.
 */
template <typename T, size_t Dimensions>
struct BlockTraits<DeviceBlock<T, Dimensions>> {
 private:
  /** Defines the allocation traits for the type. */
  using LayoutTraits = layout_traits_t<T>;

 public:
  // clang-format off
  /** Defines the value type of the block. */
  using Value     = typename LayoutTraits::Value;
  /** Defines the type of allocator for the tensor. */
  using Allocator = typename LayoutTraits::Allocator;
  /** Defines the type of the space information for the block. */
  using Space     = DynamicMultidimSpace<Dimensions>;
  /** Defines the type of the iterator for the block. */
  using Iter      = BlockIterator<Value, Space>;

  /** Defines the number of dimensions for the block. */
  static constexpr size_t dimensions      = Dimensions;
  /** Defines that the traits are for a valid block. */
  static constexpr bool   is_block        = true;
  /** Defines that the blocks is not a host block. */
  static constexpr bool   is_host_block   = false;
  /** Defines that the block is a device block */
  static constexpr bool   is_device_block = true;
  /** Defines the alignment for the block. */
  static constexpr size_t alignment       = LayoutTraits::alignment;
  // clang-format on
};

/*==--- [block enabled] ----------------------------------------------------==*/

/**
 * An interface for types which should support block functionality, which is
 * essentially an interface for providing a size and an iterator over the block.
 * \tparam Impl The implementation of the interface.
 */
template <typename Impl>
struct BlockEnabled {
  /**
   * Gets a const pointer to the implementation.
   * \return A const pointer to the implementation type.
   */
  auto impl() const noexcept -> const Impl* {
    return static_cast<const Impl*>(this);
  }

  /**
   * Gets a pointer to the implementation type.
   * \return A pointer to the implementation type.
   */
  auto impl() noexcept -> Impl* {
    return static_cast<Impl*>(this);
  }

 public:
  /**
   * Gets an iterator to the beginning of the block.
   * \return An iterator to the beginning of the block.
   */
  auto begin() noexcept -> typename BlockTraits<Impl>::Iter {
    return impl()->begin();
  }

  /**
   * Gets the stream for the block.
   * \return The stream for the block.
   */
  decltype(auto) stream() const noexcept {
    return impl()->stream();
  }

  /**
   * Returns the total number of elements for the block in the given  dimension.
   *
   * \param dim The dimension to get the size of.
   * \param Dim The type of the dimension specifier.
   * \return The number of elements in the given dimension.
   */
  template <typename Dim>
  auto size(Dim&& dim) const noexcept -> size_t {
    return impl()->size(static_cast<Dim&&>(dim));
  }
};

/**
 * Traits for block enabled type. This should be specialized for types which
 * implement the BlockEnabled interface.
 * \tparam T The type which is block enabled.
 */
template <typename T>
struct BlockEnabledTraits {
  /** The value type for the block enabled type. */
  using Value = void*;

  /** The number of dimensions for the block enabled type. */
  static constexpr size_t dimensions = 0;
};

/**
 * Specialization of the traits for a host block.
 * \tparam T    The data type for the block.
 * \tparam Dims The number of dimensions for the block.
 */
template <typename T, size_t Dims>
struct BlockEnabledTraits<HostBlock<T, Dims>> {
  /** The number of dimensions for the block enabled type. */
  static constexpr size_t dimensions = Dims;

  /** The value type for the block enabled type. */
  using Value = T;
};

/**
 * Specialization of the traits for a device block.
 * \tparam T    The data type for the block.
 * \tparam Dims The number of dimensions for the block.
 */
template <typename T, size_t Dims>
struct BlockEnabledTraits<DeviceBlock<T, Dims>> {
  /** The number of dimensions for the block enabled type. */
  static constexpr size_t dimensions = Dims;

  /** The value type for the block enabled type. */
  using Value = T;
};

/*==--- [aliases] ----------------------------------------------------------==*/

/**
 * Alias for block enabled traits with a decayed type.
 * \tparam T The type to get the block enabled traits for.
 */
template <typename T>
using block_enabled_traits_t = BlockEnabledTraits<std::decay_t<T>>;

/**
 * Defines the block traits for a type T after decaying the type T.
 * \tparam T The type to get the block traits for.
 */
template <typename T>
using block_traits_t = BlockTraits<std::decay_t<T>>;

/**
 * Alias for a 1-dimensional host block.
 * \tparam T The type of the data for the block.
 */
template <typename T>
using HostBlock1d = HostBlock<T, 1>;

/**
 * Alias for a 1-dimensional device block.
 * \tparam T The type of the data for the block.
 */
template <typename T>
using DeviceBlock1d = DeviceBlock<T, 1>;

/**
 * Alias for a 2-dimensional host block.
 * \tparam T The type of the data for the block.
 */
template <typename T>
using HostBlock2d = HostBlock<T, 2>;

/**
 * Alias for a 2-dimensional device block.
 * \tparam T The type of the data for the block.
 */
template <typename T>
using DeviceBlock2d = DeviceBlock<T, 2>;

/**
 * Alias for a 3-dimensional host block.
 * \tparam T The type of the data for the block.
 */
template <typename T>
using HostBlock3d = HostBlock<T, 3>;

/**
 * Alias for a 3-dimensional device block.
 * \tparam T The type of the data for the block.
 */
template <typename T>
using DeviceBlock3d = DeviceBlock<T, 3>;

/*==--- [traits] -----------------------------------------------------------==*/

/**
 *  Returns true if the type T is a block.
 * \tparam T The type to determine if is a block.
 */
template <typename T>
static constexpr auto is_block_v = block_traits_t<T>::is_block;

/**
 * Returns true if the type T is a host block.
 * \tparam T The type to determine if is a host block.
 */
template <typename T>
static constexpr auto is_host_block_v = block_traits_t<T>::is_host_block;

/**
 * Returns true if the type T is a device block.
 * \tparam T The type to determine if is a device block.
 */
template <typename T>
static constexpr auto is_device_block_v = block_traits_t<T>::is_device_block;

/**
 * Returns true if the type T is block enabled.
 * \tparam T The type to determine if is block enabled.
 */
template <typename T>
static constexpr bool is_block_enabled_v =
  std::is_base_of_v<BlockEnabled<std::decay_t<T>>, std::decay_t<T>>;

/**
 * Defines a valid type if T is a block.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_enable_t = std::enable_if_t<is_block_v<T>, int>;

/**
 * Defines a valid type if T is not block enabled.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_block_enable_t = std::enable_if_t<!is_block_v<T>, int>;

/**
 * Defines a valid type if T is block enabled.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_enabled_t =
  std::enable_if_t<is_block_enabled_v<T> || is_block_v<T>, int>;

/**
 * Defines a valid type if T is not block enabled.
 * tparam T The type to base the enable on.
 */
template <typename T>
using non_block_enabled_t =
  std::enable_if_t<!is_block_enabled_v<T> && !is_block_v<T>, int>;

/**
 * Defines a valid type if T is a 1 dimensional block.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_1d_enable_t =
  std::enable_if_t<is_block_v<T> && block_traits_t<T>::dimensions == 1, int>;

/**
 * Defines a valid type if T is a 2 dimensional block.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_2d_enable_t =
  std::enable_if_t<is_block_v<T> && block_traits_t<T>::dimensions == 2, int>;

/**
 * Defines a valid type if T is a 3 dimensional block.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_3d_enable_t =
  std::enable_if_t<is_block_v<T> && block_traits_t<T>::dimensions == 3, int>;

/**
 * Defines a valid type if T is a 1 dimensional block enabled type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_enabled_1d_enable_t = std::enable_if_t<
  (is_block_v<T> && block_traits_t<T>::dimensions == 1) ||
    (is_block_enabled_v<T> && block_enabled_traits_t<T>::dimensions == 1),
  int>;

/**
 * Defines a valid type if T is a 2 dimensional block enabled type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_enabled_2d_enable_t = std::enable_if_t<
  (is_block_v<T> && block_traits_t<T>::dimensions == 2) ||
    (is_block_enabled_v<T> && block_enabled_traits_t<T>::dimensions == 2),
  int>;

/**
 * Defines a valid type if T is a 3 dimensional block enabled type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using block_enabled_3d_enable_t = std::enable_if_t<
  (is_block_v<T> && block_traits_t<T>::dimensions == 3) ||
    (is_block_enabled_v<T> && block_enabled_traits_t<T>::dimensions == 3),
  int>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_BLOCK_TRAITS_HPP

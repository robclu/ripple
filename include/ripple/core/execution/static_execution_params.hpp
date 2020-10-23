//==--- ripple/core/execution/static_execution_params.hpp -- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  static_execution_params.hpp
/// \brief This file implements compile time execution parameters.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_STATIC_EXECUTION_PARAMS_HPP
#define RIPPLE_EXECUTION_STATIC_EXECUTION_PARAMS_HPP

#include "execution_params.hpp"
#include <ripple/core/iterator/block_iterator.hpp>
#include <ripple/core/multidim/static_multidim_space.hpp>
#include <ripple/core/storage/storage_traits.hpp>

namespace ripple {

/**
 * The StaticExecParams struct defines the parameters of an execution space for
 * which the size is static, and known at compile time.
 *
 * \tparam SizeX   The number of threads to execute in the x dimension.
 * \tparam SizeY   The number of threads to execute in the y dimension..
 * \tparam SizeZ   The number of threads to execute in the z dimension.
 * \tparam Padding The amount of padding for each size of each dimension.
 * \tparam Shared  A type for tile local shared memory.
 */
template <
  size_t SizeX,
  size_t SizeY,
  size_t SizeZ,
  size_t Padding,
  typename Shared>
struct StaticExecParams
: public ExecParams<StaticExecParams<SizeX, SizeY, SizeZ, Padding, Shared>> {
 private:
  /*==--- [aliases] --------------------------------------------------------==*/

  // clang-format off
  /** Defines the layout traits for the shared memory type. */
  using Traits    = layout_traits_t<Shared>;
  /** Defines the value type for the iterator over the execution space. */
  using Value     = typename Traits::Value;
  /** Defines the allocator type for the execution space. */
  using Allocator = typename Traits::Allocator;


  /** Defines the type of the 1D space. */
  using Space1d = StaticMultidimSpace<SizeX>;
  /** Defines the type of the 2D space. */
  using Space2d = StaticMultidimSpace<SizeX, SizeY>;
  /** Defines the type of the 3D space. */
  using Space3d = StaticMultidimSpace<SizeX, SizeY, SizeZ>;

  /**
   * Defines the type of the multidimensional space for the execution.
   * \tparam Dims The number of dimensions to get the space for.
   */
  template <size_t Dims>
  using Space = std::conditional_t<
    Dims == 1, Space1d, std::conditional_t<Dims == 2, Space2d, Space3d>>;

  /**
   * Defines the type of the iterator over the execution space.
   * \tparam Dims The number of dimensions for the iterator.
   */
  template <size_t Dims>
  using Iter = BlockIterator<Value, Space<Dims>>;
  // clang-format on

 public:
  /**
   * Computes the total size of the execution space, for Dims dimensions.
   * \tparam Dims The number of dimensions to get the size for.
   * \return The total number of elements in the space, including padding.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    constexpr size_t dim_pad     = Padding * 2;
    constexpr size_t dim_1d_size = SizeX + dim_pad;
    constexpr size_t dim_2d_size = dim_1d_size * (SizeY + dim_pad);
    constexpr size_t dim_3d_size = dim_2d_size * (SizeZ + dim_pad);
    return Dims == 1 ? dim_1d_size
                     : Dims == 2 ? dim_2d_size : Dims == 3 ? dim_3d_size : 0;
  }

  /**
   * Computes the size of the space in the given dimension, including padding.
   * \param  dim The dimension to get the size of the space for.
   * \tparam Dim The type of the dimension specifier.
   * \return The number of elements in the dimension, including padding.
   */
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const noexcept -> size_t {
    return size_impl(static_cast<Dim&&>(dim));
  }

  /**
   * Gets the amount of padding for the execution space.
   * \return The amount of padding of a single size of a single dimension in
   *         the space.
   */
  ripple_host_device constexpr auto padding() const noexcept -> size_t {
    return Padding;
  }

  /**
   * Gets an iterator over a memory space pointed to by data pointer, for the
   * number of specified dimensions.
   *
   * \param  data A pointer to the memory space data.
   * \tparam Dims The number of dimensions for the iterator.
   * \tparam T    The type of the data.
   * \return An iterator over the memory space.
   */
  template <size_t Dims, typename T>
  ripple_host_device auto
  iterator(T* data) const noexcept -> BlockIterator<Value, Space<Dims>> {
    using SpaceType      = Space<Dims>;
    using Iterator       = BlockIterator<Value, SpaceType>;
    constexpr auto space = SpaceType{Padding};
    return Iterator{Allocator::create(data, space), space};
  }

  /**
   * Gets the number of bytes required to allocator data for the space.
   * \tparam Dims The number of dimensions to allocate for.
   * \return The number of bytes required to allocate data for the space.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto allocation_size() const noexcept -> size_t {
    return Allocator::allocation_size(size<Dims>());
  }

 private:
  /**
   * Implementation to return the size of the execution space in the x
   * dimension.
   * \return The size of the space in the x dimension.
   */
  ripple_host_device constexpr auto size_impl(dimx_t) const noexcept -> size_t {
    return SizeX;
  }

  /**
   * Implementation to return the size of the execution space in the y
   * dimension.
   * \return The size of the space in the y dimension.
   */
  ripple_host_device constexpr auto size_impl(dimy_t) const noexcept -> size_t {
    return SizeY;
  }

  /**
   * Implementation to return the size of the execution space in the z
   * dimension.
   * \return The size of the space in the z dimension.
   */
  ripple_host_device constexpr auto size_impl(dimz_t) const noexcept -> size_t {
    return SizeZ;
  }
};

/**
 * Creates default static execution execution parameters.
 * \tparam Dims    The number of dimensions for the parameters.
 * \tparam Shared  The type of the shared memory for the parameters.
 * \tparam Pad     The padding for the space.
 */
template <size_t Dims, typename Shared = VoidShared, size_t Pad = 0>
ripple_host_device auto
static_params() noexcept -> default_shared_exec_params_t<Dims, Shared, Pad> {
  return default_shared_exec_params_t<Dims, Shared, Pad>();
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_STATIC_EXECUTION_PARAMS_HPP

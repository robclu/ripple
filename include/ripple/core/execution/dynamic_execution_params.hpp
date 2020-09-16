//==--- ripple/core/execution/dynamic_execution_params.hpp - -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dynamic_execution_params.hpp
/// \brief This file implements dynamic execution parameters.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_DYNAMIC_EXECUTION_PARAMS_HPP
#define RIPPLE_EXECUTION_DYNAMIC_EXECUTION_PARAMS_HPP

#include "execution_params.hpp"
#include <ripple/core/iterator/block_iterator.hpp>
#include <ripple/core/multidim/dynamic_multidim_space.hpp>
#include <ripple/core/storage/storage_traits.hpp>

namespace ripple {

/**
 * Thisstruct defines the parameters of an execution space for which the size
 * is dynamic. It can be used when the domain size is not determined until
 * runtime.
 *
 * \tparam Shared  A type for tile local shared memory.
 */
template <typename Shared>
struct DynamicExecParams : public ExecParams<DynamicExecParams<Shared>> {
 private:
  /*==--- [aliases] --------------------------------------------------------==*/

  // clang-format off
  /** Defines the layout traits for the shared memory type. */
  using traits_t    = layout_traits_t<Shared>;
  /** Defines the value type for the iterator over the execution space. */
  using value_t     = typename traits_t::value_t;
  /** Defines the allocator type for the execution space. */
  using allocator_t = typename traits_t::allocator_t;
  /** Defines the type of the default space for the execution. */
  using space_t     = DynamicMultidimSpace<3>;
  /** Defines the type used by the space for the step information. */
  using step_t      = typename space_t::step_t;

  /** 
   * Defines the type of the multidimensional space for the execution.
   * \tparam Dims The number of dimensions for the space.
   */
  template <size_t Dims>
  using make_space_t = DynamicMultidimSpace<Dims>;
  // clang-format on

 public:
  /*==--- [constructor] ----------------------------------------------------==*/

  /**
   * Default constructor to create a space with default sizes.
   */
  ripple_host_device constexpr DynamicExecParams() noexcept
  : _space{1024, 1, 1} {}

  /**
   * Creates the execution space without padding.
   * \param  sizes The sizes of the execution space.
   * \tparam Sizes The types of the sizes of the execution space.
   */
  template <typename... Sizes, all_arithmetic_size_enable_t<3, Sizes...> = 0>
  ripple_host_device constexpr DynamicExecParams(Sizes&&... sizes) noexcept
  : _space{static_cast<step_t>(sizes)...} {}

  /**
   * Creates the eecution space without \p padding.
   *
   * \note The padding is the amount of padding per side of each dimension, so
   *       a padding value of 2 results in 4 additional cells in each dimension,
   *       2 on each end of the dimension.
   *
   * \param  padding The amount of padding for each side of each dimension.
   * \param  sizes   The sizes of each dimension of the space.
   * \tparam Sizes   The types of the sizes.
   */
  template <typename... Sizes, all_arithmetic_size_enable_t<3, Sizes...> = 0>
  ripple_host_device constexpr DynamicExecParams(
    uint32_t padding, Sizes&&... sizes) noexcept
  : _space{padding, static_cast<step_t>(sizes)...} {}

  /*==--- [size] -----------------------------------------------------------==*/

  /**
   * Gets the total size of the execution space for the first Dims dimensions.
   *
   * \note The size here refers to the number of elements.
   *
   * \tparam Dims The number of dimensions to get the size for.
   * \return The number of elements for Dims dimensions.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto size() const noexcept -> step_t {
    static_assert(
      Dims <= 3, "Execution space can't be more than 3 dimensions!");
    auto total_size = _space.size(dim_x);
    unrolled_for<Dims - 1>([&](auto d) {
      constexpr auto dim = d + 1;
      total_size *= _space.size(dim);
    });
    return total_size;
  }

  /**
   * Gets the total size of the execution space, for Dims dimensions, with a
   * padding of \p padding.
   *
   * \note The size here refers to the number of elements.
   *
   * \param  padding The amount of padding for each side of each dimension.
   * \tparam Dims    The number of dimensions to get the size for.
   * \return The total number of elements, including padding.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto
  size(size_t padding) const noexcept -> step_t {
    static_assert(
      Dims <= 3, "Execution space can't be more than 3 dimensions!");
    const size_t pad        = padding * 2;
    auto         total_size = _space.internal_size(dim_x) + pad;
    unrolled_for<Dims - 1>([&](auto d) {
      constexpr auto dim = d + 1;
      total_size *= _space.internal_size(dim) + pad;
    });
    return total_size;
  }

  /**
   * Returns the size of the space in the \p dim dimension, without the
   * Gets the size of the space (number of elements) in the \p dim dimension,
   * without padding.
   * \param  dim The dimension to get the size of.
   * \tparam Dim The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const noexcept -> size_t {
    return _space.internal_size(std::forward<Dim>(dim));
  }

  /*==--- [properties] -----------------------------------------------------==*/

  /**
   * Gets the amount of padding for the execution space.
   *
   * \note This returns the amount of padding on a single side of a single
   * dimension.
   *
   * \return The amount of padding for the space.
   */
  ripple_host_device constexpr auto padding() const noexcept -> size_t {
    return _space.padding();
  }

  /*==--- [creation] -------------------------------------------------------==*/

  /**
   * Returns an iterator over a memory space pointed to by the \p data, which
   * will iterate over a space defined by this space.
   * \param  data The data to get an iterator over.
   * \tparam Dims The number of dimension to get an iterator for.
   * \tparam T    The type of the data pointed to.
   * \return A block iterator over a space defined by this space.
   */
  template <size_t Dims, typename T>
  ripple_host_device auto iterator(T* data) const noexcept
    -> BlockIterator<value_t, make_space_t<Dims>> {
    using _space_t = make_space_t<Dims>;
    using iter_t   = BlockIterator<value_t, _space_t>;
    _space_t space;
    unrolled_for<Dims>([&](auto d) {
      constexpr auto dim = d;
      space[dim]         = _space[dim];
    });
    space.padding() = _space.padding();
    return iter_t{allocator_t::create(data, space), space};
  }

  /**
   * Gets the number of bytes required to allocator data for the space.
   * \tparam Dims The number of dimensions to allocate for.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto allocation_size() const noexcept -> size_t {
    return allocator_t::allocation_size(size<Dims>());
  }

 private:
  space_t _space; //!< The execution space.
};

/*==--- [functions] --------------------------------------------------------==*/

/**
 * Creates a default dynamic execution execution paramters for the device.
 * \param  padding The amount of padding for the space.
 * \tparam Dims    The number of dimensions for the parameters.
 * \tparam Shared  The type of the shared memory for the parameters.
 */
template <size_t Dims, typename Shared = VoidShared>
ripple_host_device auto
dynamic_params(size_t padding = 0) noexcept -> DynamicExecParams<Shared> {
  constexpr auto size_x = (Dims == 1 ? 512 : Dims == 2 ? 32 : 8);
  constexpr auto size_y = (Dims == 1 ? 1 : Dims == 2 ? 16 : 8);
  constexpr auto size_z = (Dims == 1 ? 1 : Dims == 2 ? 1 : 4);

  return DynamicExecParams<Shared>(padding, size_x, size_y, size_z);
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_DYNAMIC_EXECUTION_PARAMS_HPP

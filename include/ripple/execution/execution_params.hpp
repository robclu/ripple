/**=--- ripple/execution/execution_params.hpp -------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  execution_params.hpp
 * \brief This file defines an interface for execution parameters.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP
#define RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP

#include "execution_traits.hpp"
#include <ripple/utility/dim.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/**
 * The ExecParams struct defines a static interface for all execution parameter
 * implementations. The purpose of execution paramters are to customize the
 * execution space and to efficiently create data and iterators over the data
 * in the executin space for different configurations.
 *
 * \tparam Impl The implementation of the interface.
 */
template <typename Impl>
struct ExecParams {
 private:
  /** Defines the traits for the implementation. */
  using Traits = ExecTraits<Impl>;

  /**
   * Gets a const pointer to the implementation.
   * \return A const pointer to the implementation.
   */
  ripple_host_device constexpr auto impl() const noexcept -> const Impl* {
    return static_cast<const Impl*>(this);
  }

  /**
   * Gets a pointer to the implementation.
   * \return A pointer to the implementation.
   */
  ripple_host_device constexpr auto impl() noexcept -> Impl* {
    return static_cast<Impl*>(this);
  }

 public:
  /**
   * Gets the total size of the execution space, for the given number of
   * dimensions.
   * \tparam Dims The number of dimensions to get the size for.
   * \return The total number of elements in the execution space.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto size() const noexcept {
    return impl()->template size<Dims>();
  }

  /**
   * Gets the total size of the execution space, for Dims dimensions, with the
   * given padding amount.
   *
   * \note The size here refers to the number of elements.
   *
   * \param  padding The amount of padding for each side of each dimension.
   * \tparam Dims    The number of dimensions to get the size for.
   * \return The total number of elements, including padding.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto size(size_t padding) const noexcept {
    return impl()->template size<Dims>(padding);
  }

  /**
   * Gets the size of the space in the given dimension.
   * \param  dim The dimension to get the size of the space for.
   * \tparam Dim The type of the dimension specifier.
   * \return The size of the given dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const noexcept -> size_t {
    return impl()->size(ripple_forward(dim));
  }

  /**
   * Gets the amount of padding for the execution space, per side of each
   * dimension.
   * \return The amount of padding for a side of a dimension in the space.
   */
  ripple_host_device constexpr auto padding() const noexcept -> size_t {
    return impl()->padding();
  }

  /**
   * Gets an iterator over a memory space pointed to by the data, for a
   * given number of dimensions.
   * \param  data A pointer to the memory space data.
   * \tparam Dims The number of dimensions for the iterator.
   * \tparam T    The type of the data to create the iterator over.
   * \return An iterator over the execution space which iterates over elements
   *        of the given type.
   */
  template <size_t Dims, typename T>
  ripple_host_device decltype(auto) iterator(T* data) const noexcept {
    return impl()->template iterator<Dims>(data);
  }

  /**
   * Gets the number of bytes required to allocator data for the space, for the
   * given number of dimensions.
   * \tparam Dims The number of dimensions to allocate for.
   */
  template <size_t Dims>
  ripple_host_device constexpr auto allocation_size() const noexcept -> size_t {
    return impl()->template allocation_size<Dims>();
  }
};

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP

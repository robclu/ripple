//==--- ripple/core/multidim/multidim_space.hpp ------------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multidim_space.hpp
/// \brief This file defines a static interface for multidimensional spaces.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MULTIDIM_MULTIDIM_SPACE_HPP
#define RIPPLE_MULTIDIM_MULTIDIM_SPACE_HPP

#include "space_traits.hpp"
#include <ripple/core/utility/forward.hpp>
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

/**
 * The MultidimSpace defines an interface for classes when define a
 * multidimensional space.
 * \tparam Impl The implementation of the interface.
 */
template <typename Impl>
struct MultidimSpace {
 private:
  /**
   * Gets a const pointer to the implementation.
   * \return A const pointer to the implementation.
   */
  ripple_host_device constexpr auto impl() const noexcept -> const Impl* {
    return static_cast<const Impl*>(this);
  }

  /**
   * Gets a pointer to the implemenation.
   * \return A pointer to the implementation.
   */
  ripple_host_device constexpr auto impl() noexcept -> Impl* {
    return static_cast<Impl*>(this);
  }

 public:
  /**
   * Gets a reference to the amount of padding for each side of each
   * dimension in the space.
   * \return A reference to the amount of padding for a side of the dimension.
   */
  ripple_host_device constexpr auto padding() noexcept -> size_t& {
    return impl()->padding();
  }

  /**
   * Gets the amount of padding for each side of each dimension in the
   * space.
   * \return The amount of padding per side of the size.
   */
  ripple_host_device constexpr auto padding() const noexcept -> size_t {
    return impl()->padding();
  }

  /**
   * Gets the total amount of padding for each dimension, which is the sum
   * of the padding on each side of the dimension.
   * \return The amount of padding per dimension.
   */
  ripple_host_device constexpr auto dim_padding() const noexcept -> size_t {
    return impl()->dim_padding();
  }

  /**
   * Gets the number of dimensions in the space.
   * \return The number of dimensions in the space.
   */
  ripple_host_device constexpr auto dimensions() const noexcept -> size_t {
    return SpaceTraits<Impl>::dimensions;
  }

  /**
   * Gets the size of the given dimension, including the padding.
   * \param  dim  The dimension to get the size of.
   * \tparam Dim  The type of the dimension.
   * \return The total number of elements in the dimension, including padding.
   */
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const noexcept -> size_t {
    return impl()->size(ripple_forward(dim));
  }

  /**
   * Gets the total size of the N dimensional space i.e the total number of
   * elements in the space, including the padding for each of the dimensions.
   * \return The total number of elements in the space, including padding
   *         elements.
   */
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return impl()->size();
  }

  /**
   * Gets the size of the given dimension, excluding padded elements.
   * \param  dim  The dimension to get the size oAf.
   * \tparam Dim  The type of the dimension.
   * \return The number of elements in the dimension, excluding padding
   *         elements.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  internal_size(Dim&& dim) const noexcept -> size_t {
    return impl()->internal_size(ripple_forward(dim));
  }

  /**
   * Gets the total internal size of the N dimensional space i.e the total
   * number of elements in the space, exluding any padding elements.
   */
  ripple_host_device constexpr auto internal_size() const noexcept -> size_t {
    return impl()->internal_size();
  }

  /**
   * Gets the step size to from one element in given dimension to the next
   * element in the same dimension.
   * \param  dim   The dimension to get the step size in.
   * \tparam Dim   The type of the dimension.
   * \return The step size between successive elements in the same dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto step(Dim&& dim) const noexcept -> size_t {
    return impl()->step(ripple_forward(dim));
  }
};

} // namespace ripple

#endif // RIPPLE_MULTIDIM_MULTIDIM_SPACE_HPP

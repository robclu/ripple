//==--- ripple/core/multidim/static_multidim_space.hpp ----- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  static_multidim_space.hpp
/// \brief This file imlements a class which defines a statically sized multi
///        dimensional space.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MULTIDIM_STATIC_MULTIDIM_SPACE_HPP
#define RIPPLE_MULTIDIM_STATIC_MULTIDIM_SPACE_HPP

#include "multidim_space.hpp"

namespace ripple {

/**
 * The StaticMultidimSpace struct defines spatial information over multiple
 * dimensions. Specifically, it defines the sizes of the dimensions and the
 * steps required to get from one element to another in a given dimensions of
 * the space. The static nature of the space means that it can't be modified,
 * and the size and  steps for the space are all known at compile time, which
 * makes using this space more efficient, all be it with less flexibility.
 * \tparam Sizes The sizes of the dimensions of the space.
 */
template <size_t... Sizes>
struct StaticMultidimSpace
: public MultidimSpace<StaticMultidimSpace<Sizes...>> {
 private:
  /** Defines the number of dimension for the space. */
  static constexpr size_t dims = sizeof...(Sizes);

  size_t padding_ = 0; //!< The amount of padding for the space.

 public:
  /** Default constructor. */
  ripple_host_device constexpr StaticMultidimSpace() = default;

  /**
   * Constructor to set the padding for the space.
   * \param padding The amount of padding for the space.
   */
  ripple_host_device constexpr StaticMultidimSpace(size_t padding) noexcept
  : padding_{padding} {}

  /**
   * Gets the amount of padding for a single side of a single dimension
   * of the space.
   * \return The amount of padding for the space.
   */
  ripple_host_device constexpr auto padding() noexcept -> size_t& {
    return padding_;
  }

  /**
   * Gets the amount of padding fot a single size of a single dimension of
   * the space.
   * \return The amount of padding for the space.
   */
  ripple_host_device constexpr auto padding() const noexcept -> size_t {
    return padding_;
  }

  /**
   *  Gets the total amount of padding for the dimension, which is twice the
   *  padding amount of each side.
   *
   * \sa padding
   *
   * \return The total padding amount for both sides of the dimension.
   */
  ripple_host_device constexpr auto dim_padding() const noexcept -> size_t {
    return padding_ * 2;
  }

  /**
   * Gets the number of dimensions for the space.
   * \return The number of dimensions for the space.
   */
  ripple_host_device constexpr auto dimensions() const noexcept -> size_t {
    return dims;
  }

  /**
   * Gets the total size of the N dimensional space i.e the total number of
   * elements in the space, including the padding for the space.
   *
   * \note This is the product sum of the dimension sizes including padding.
   *
   * \return the total number of elements in the space including padding.
   */
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return ((Sizes + dim_padding()) * ... * size_t{1});
  }

  /**
   * Gets the size of the dimension with padding.
   * \param  dim The dimension to get the size for.
   * \tparam Dim The type of the dimension specifier.
   * \return The number of elements in the dimension, including padding.
   */
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const noexcept -> size_t {
    constexpr size_t sizes[dims] = {Sizes...};
    return sizes[dim] + dim_padding();
  }

  /**
   * Gets the size of the given dimension, without padding.
   * \param  dim  The dimension to get the size of.
   * \tparam Dim  The type of the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  internal_size(Dim&& dim) const noexcept -> size_t {
    constexpr size_t sizes[dims] = {Sizes...};
    return sizes[dim];
  }

  /**
   * Computes the total size of the N dimensional space without paddding, which
   * is the total number of computable elements in the space.
   * \return The total number of elements in the space, excluding padding.
   */
  ripple_host_device constexpr auto internal_size() const noexcept -> size_t {
    return (Sizes * ... * size_t{1});
  }

  /**
   * Gets the step size to from one element in the given dimension to the next
   * element in same dimension.
   * \param  dim   The dimension to get the step size in.
   * \tparam Dim   The type of the dimension.
   * \return The step size between two elements in the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto step(Dim&& dim) const noexcept -> size_t {
    constexpr size_t sizes[dims] = {Sizes...};
    size_t           res         = 1;
    for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
      res *= sizes[i] + dim_padding();
    }
    return res;
  }
};

} // namespace ripple

#endif // RIPPLE_MULTIDIM_STATIC_MULTIDIM_SPACE_HPP

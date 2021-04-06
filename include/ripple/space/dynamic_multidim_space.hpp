/**=--- ripple/space/dynamic_multidim_space.hpp ------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  dynamic_multidim_space.hpp
 * \brief This file imlements a class which defines a dynamic multidimensional
 *        space
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_SPACE_DYNAMIC_MULTIDIM_SPACE_HPP
#define RIPPLE_SPACE_DYNAMIC_MULTIDIM_SPACE_HPP

#include "multidim_space.hpp"
#include <ripple/container/vec.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple {

/**
 * The DynamicMultidimSpace struct defines spatial information over multiple
 * dimensions, specifically the sizes of the dimensions and the steps required
 * to get from one element to another in the dimensions of the space. The
 * dynamic nature of the space means that it can be modified. It implements the
 * MultidimSpace interface.
 * \tparam Dimensions The number of dimensions.
 */
template <size_t Dimensions>
struct DynamicMultidimSpace
: public MultidimSpace<DynamicMultidimSpace<Dimensions>> {
  // clang-format off
  /** Defines the size type used for padding. */
  using Padding = uint32_t;
  /** Defines the type used to store step information. */
  using Step    = uint32_t;
  // clang-format on

 private:
  /** Defines the number of dimensions */
  static constexpr size_t dims = Dimensions;

  /** Defines the type of container to store the size and step information. */
  using Container = Vec<Step, dims>;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /**
   * Default constructor -- enables creation of empty spatial information.
   */
  constexpr DynamicMultidimSpace() = default;

  /**
   * Sets the sizes of the dimensions for the space.
   * \param  sizes The sizes of the dimensions.
   * \tparam Sizes The type of the sizes.
   */
  template <typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> = 0>
  ripple_host_device constexpr DynamicMultidimSpace(Sizes&&... sizes) noexcept
  : sizes_{static_cast<Step>(sizes)...} {}

  /**
   * Sets the sizes of the dimensions for the space and the amount of padding
   * for the space.
   * \param  padding The amount of padding for a side of each dimension.
   * \param  sizes   The sizes of the dimensions.
   * \tparam Sizes   The type of the sizes.
   */
  template <typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> = 0>
  ripple_host_device constexpr DynamicMultidimSpace(
    Padding padding, Sizes&&... sizes) noexcept
  : sizes_{static_cast<Step>(sizes)...}, padding_{padding} {}

  /**
   * Gets the amount of padding for each side of each dimension for the
   * space.
   * \return A reference to the amount of padding on one side of each dimension.
   */
  ripple_host_device constexpr auto padding() noexcept -> Padding& {
    return padding_;
  }

  /**
   * Gets the amount of padding for each side of each dimension for the
   * space.
   * \return The amount of padding on one side of each dimension.
   */
  ripple_host_device constexpr auto padding() const noexcept -> Padding {
    return padding_;
  }

  /**
   * Gets the total amounnt of padding for the dimesion, which is twice the
   * dimension per side.
   * \return The total amount of padding for a dimension.
   */
  ripple_host_device constexpr auto dim_padding() const noexcept -> Padding {
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
   * Resizes each of the dimensions specified by the \p size. If
   * `sizeof...(Sizes) < dims`, the first `sizeof...(Sizes)` dimensions are
   * resized. If `sizeof...(Sizes) > dims`, then a compile time error is
   * generated.
   *
   * \param  sizes The sizes to resize the dimensions to.
   * \tparam Sizes The type of the sizes.
   */
  template <typename... Sizes>
  ripple_host_device auto resize(Sizes&&... sizes) noexcept -> void {
    constexpr size_t num_sizes = sizeof...(Sizes);
    static_assert(num_sizes <= dims, "Too many sizes specified in resize.");

    const Step dim_sizes[num_sizes] = {static_cast<Step>(sizes)...};
    unrolled_for<num_sizes>([&](auto i) { sizes_[i] = dim_sizes[i]; });
  }

  /**
   * Resizes the given dimension to have size number of elements.
   * \param  dim  The dimension to resize.
   * \param  size The size to resize the dimension to.
   * \tparam Dim  The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_host_device auto resize_dim(Dim&& dim, Step size) noexcept -> void {
    sizes_[dim] = size;
  }

  /**
   * Gets the size of the given dimension.
   * \param  dim  The dimension to get the size of.
   * \tparam Dim  The type of the dimension.
   * \return The number of elements for the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const noexcept -> Step {
    return sizes_[dim] + dim_padding();
  }

  /**
   * Gets the total size of the N dimensional space i.e the total number of
   * elements in the space. This is the product sum of the dimension sizes,
   * *including* the padding for the space.
   * \return The total number of elements in the space, including padding.
   */
  ripple_host_device constexpr auto size() const noexcept -> Step {
    Step prod_sum = 1;
    unrolled_for<dims>(
      [&](auto dim) { prod_sum *= (sizes_[dim] + dim_padding()); });
    return prod_sum;
  }

  /**
   * Gets the internals size of the given dimension -- the size without padding
   * elements.
   * \param  dim  The dimension to get the size of.
   * \tparam Dim  The type of the dimension.
   * \return The number of elements in the given dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  internal_size(Dim&& dim) const noexcept -> Step {
    return sizes_[dim];
  }

  /**
   * Gets the total internal  size of the N dimensional space i.e the total
   * number of elements in the space.
   *
   * \note This is the product sum of the dimension sizes, *not including*
   *       padding.
   *
   * \return The total number of internal elements for the space.
   */
  ripple_host_device constexpr auto internal_size() const noexcept -> Step {
    Step prod_sum = 1;
    unrolled_for<dims>([&](auto dim) { prod_sum *= sizes_[dim]; });
    return prod_sum;
  }

  /**
   * Returns the step size to from one element in the given dimension to the
   * next element in the given dimension.
   *
   * \param  dim The dimension to get the step size in.
   * \tparam Dim The type of the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto step(Dim&& dim) const noexcept -> Step {
    using DimType = std::decay_t<Dim>;
    Step res      = 1;
    if constexpr (is_dimension_v<DimType>) {
      constexpr size_t end = static_cast<size_t>(DimType::value);
      unrolled_for<end>([&](auto d) { res *= (sizes_[d] + dim_padding()); });
    } else {
      for (size_t d = 0; d < static_cast<size_t>(dim); ++d) {
        res *= sizes_[d] + dim_padding();
      }
    }
    return res;
  }

  /**
   * Gets a reference to the size of the given dimension.
   * \param  dim The dimension size to get a refernece to.
   * \tparam Dim The type of the dimension specifier.
   * \return A reference to the size of the space for the given dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto operator[](Dim&& dim) noexcept -> Step& {
    return sizes_[dim];
  }

  /**
   * Gets a constt reference to the size of the given dimension.
   * \param  dim The dimension size to get a refernece to.
   * \tparam Dim The type of the dimension specifier.
   * \return A const reference to the size of the space for the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  operator[](Dim&& dim) const noexcept -> const Step& {
    return sizes_[dim];
  }

 private:
  Container sizes_;       //!< Sizes of the dimensions.
  Padding   padding_ = 0; //!< Amount of padding for each side of each dim.
};

} // namespace ripple

#endif // RIPPLE_SPACE_MULTIDIM_SPACE_HPP

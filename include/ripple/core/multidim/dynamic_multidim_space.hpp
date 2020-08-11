//==--- ripple/core/multidim/dynamic_multidim_space.hpp --------- -*- C++ -*-
//---==//
//
//                                Ripple
//
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dynamic_multidim_space.hpp
/// \brief This file imlements a class which defines a dynamic multidimensional
///        space
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MULTIDIM_DYNAMIC_MULTI_SPACE_HPP
#define RIPPLE_MULTIDIM_DYNAMIC_MULTI_SPACE_HPP

#include "multidim_space.hpp"
#include <ripple/core/container/vec.hpp>
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

/// The DynamicMultidimSpace struct defines spatial information over multiple
/// dimensions, specifically the sizes of the dimensions and the steps required
/// to get from one element to another in the dimensions of the space. The
/// dynamic nature of the space means that it can be modified. It implements the
/// MultidimSpace interface.
/// \tparam Dimensions The number of dimensions.
template <std::size_t Dimensions>
struct DynamicMultidimSpace
: public MultidimSpace<DynamicMultidimSpace<Dimensions>> {
  // clang-format off
  /// Defines the size type used for padding.
  using padding_t   = uint32_t;
  /// Defines the type used to store step information.
  using step_t      = uint32_t;
  // clang-format on

 private:
  /// Defines the number of dimensions
  static constexpr size_t dims = Dimensions;

  /// Defines the type of container to store the size and step information.
  using container_t = Vec<step_t, dims>;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor -- enables creation of empty spatial information.
  constexpr DynamicMultidimSpace() = default;

  /// Sets the sizes of the dimensions for the space.
  /// \param  sizes The sizes of the dimensions.
  /// \tparam Sizes The type of the sizes.
  template <typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> = 0>
  ripple_host_device constexpr DynamicMultidimSpace(Sizes&&... sizes)
  : _sizes{static_cast<step_t>(sizes)...} {}

  /// Sets the sizes of the dimensions for the space and the amount of padding
  /// for the space.
  /// \param  padding The amount of padding for each of the sides of each of the
  ///                 dimensions.
  /// \param  sizes   The sizes of the dimensions.
  /// \tparam Sizes   The type of the sizes.
  template <typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> = 0>
  ripple_host_device constexpr DynamicMultidimSpace(
    padding_t padding, Sizes&&... sizes)
  : _padding{padding}, _sizes{static_cast<step_t>(sizes)...} {}

  //==--- [padding] --------------------------------------------------------==//

  /// Returns the amount of padding for each side of each dimension for the
  /// space.
  ripple_host_device constexpr auto padding() -> padding_t& {
    return _padding;
  }

  /// Returns the amount of padding for each side of each dimension for the
  /// space.
  ripple_host_device constexpr auto padding() const -> padding_t {
    return _padding;
  }

  /// Returns the total amounnt of padding for the dimesion, which is twice the
  /// dimension per side.
  ripple_host_device constexpr auto dim_padding() const -> padding_t {
    return _padding * 2;
  }

  //==--- [size] -----------------------------------------------------------==//

  /// Returns the number of dimensions for the space.
  ripple_host_device constexpr auto dimensions() const -> size_t {
    return dims;
  }

  /// Resizes each of the dimensions specified by the \p size. If
  /// sizeof...(Sizes) < dims, the first sizeof...(Sizes) dimensions are
  /// resized. If sizeof...(Sizes) > dims, then a compile time error is
  /// generated.
  /// \param  sizes The sizes to resize the dimensions to.
  /// \tparam Sizes The type of the sizes.
  template <typename... Sizes>
  ripple_host_device auto resize(Sizes&&... sizes) -> void {
    constexpr auto num_sizes = sizeof...(Sizes);
    static_assert(num_sizes <= dims, "Too many sizes specified in resize.");

    const step_t dim_sizes[num_sizes] = {static_cast<step_t>(sizes)...};
    unrolled_for<num_sizes>([&](auto i) { _sizes[i] = dim_sizes[i]; });
  }

  /// Resizes the \p dim dimensions to have \p size.
  /// \param  dim  The dimension to resize.
  /// \param  size The size to resize the dimension to.
  /// \tparam Dim  The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto resize_dim(Dim&& dim, step_t size) -> void {
    _sizes[dim] = size;
  }

  /// Returns the size of the \p dim dimension.
  /// \param  dim  The dimension to get the size of.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> step_t {
    return _sizes[dim] + dim_padding();
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension sizes,
  /// with the padding for the space.
  ripple_host_device constexpr auto size() const -> step_t {
    step_t prod_sum = 1;
    unrolled_for<dims>(
      [&](auto dim) { prod_sum *= (_sizes[dim] + dim_padding()); });
    return prod_sum;
  }

  /// Returns the size of the \p dim dimension, without padding.
  /// \param  dim  The dimension to get the size of.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto internal_size(Dim&& dim) const -> step_t {
    return _sizes[dim];
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension sizes,
  /// without the padding for the dimensions.
  ripple_host_device constexpr auto internal_size() const -> step_t {
    step_t prod_sum = 1;
    unrolled_for<dims>([&](auto dim) { prod_sum *= _sizes[dim]; });
    return prod_sum;
  }

  //==--- [step] -----------------------------------------------------------==//

  /// Returns the step size to from one element in \p dim to the next element in
  /// \p dim.
  /// \param  dim   The dimension to get the step size in.
  /// \param  width The width of the array if the space is for soa or soa.
  /// \tparam Dim   The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto step(Dim&& dim) const -> step_t {
    using dim_t = std::decay_t<Dim>;
    step_t res  = 1;
    if constexpr (is_dimension_v<dim_t>) {
      constexpr size_t end = static_cast<size_t>(dim_t::value);
      unrolled_for<end>([&](auto d) { res *= (_sizes[d] + dim_padding()); });
    } else {
      for (size_t d : range(static_cast<size_t>(dim))) {
        res *= _sizes[d] + dim_padding();
      }
    }
    return res;
  }

  //==--- [access] ---------------------------------------------------------==//

  /// Returns a reference to the size of dimension \p dim, which can be used to
  /// set the size of the dimension.
  /// \param  dim The dimension size to get a refernece to.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto operator[](Dim&& dim) -> step_t& {
    return _sizes[dim];
  }

  /// Returns a cosnt reference to the size of dimension \p dim, which can be
  /// used to detemine the size of one of the dimensions in the space.
  /// \param  dim The dimension size to get a refernece to.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto
  operator[](Dim&& dim) const -> const step_t& {
    return _sizes[dim];
  }

 private:
  padding_t   _padding = 0; //!< Amount of padding for each side of each dim.
  container_t _sizes;       //!< Sizes of the dimensions.
};

} // namespace ripple

#endif // RIPPLE_MULTIDIM_MULTIDIM_SPACE_HPP

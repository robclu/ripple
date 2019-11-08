//==--- ripple/multidim/dynamic_multidim_space.hpp --------- -*- C++ -*- ---==//
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
#include <ripple/container/vec.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple {

/// The DynamicMultidimSpace struct defines spatial information over multiple
/// dimensions, specifically the sizes of the dimensions and the steps required
/// to get from one element to another in the dimensions of the space. The
/// dynamic nature of the space means that it can be modified. It implements the
/// MultidimSpace interface.
/// \tparam Dimensions The number of dimensions.
template <std::size_t Dimensions>
struct DynamicMultidimSpace :
  public MultidimSpace<DynamicMultidimSpace<Dimensions>> {
 private:
  /// Defines the number of dimensions
  static constexpr auto dims = Dimensions;

  /// Defines the type of container to store the size and step information.
  using container_t = Vec<std::size_t, dims>;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor -- enables creation of empty spatial information.
  DynamicMultidimSpace() = default;

  /// Sets the sizes of the dimensions for the space.
  /// \param  sizes The sizes of the dimensions. 
  /// \tparam Sizes The type of the sizes.
  template <typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> = 0>
  ripple_host_device DynamicMultidimSpace(Sizes&&... sizes)
  : _sizes{static_cast<std::size_t>(sizes)...} {}

  //==--- [padding] --------------------------------------------------------==//

  /// Returns the amount of padding in dimension \p dim. This implementation of
  /// the multidimensional space has no padding.
  /// \param[in] dim The dimension to get the padding for.
  template <typename Dim>
  ripple_host_device constexpr auto padding(Dim&& dim) -> std::size_t {
    return 0;
  }

  //==--- [size] -----------------------------------------------------------==//
  
  /// Returns the number of dimensions for the space.
  ripple_host_device constexpr auto dimensions() const -> std::size_t {
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
    static_assert(
      num_sizes <= dims, "Too many sizes specified in resize."
    );

    const std::size_t dim_sizes[num_sizes] = { 
      static_cast<std::size_t>(sizes)...
    };
    unrolled_for<num_sizes>([&] (auto i) {
      _sizes[i] = dim_sizes[i];
    });
  }

  /// Returns the size of the \p dim dimension.
  /// \param  dim  The dimension to get the size of.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device auto size(Dim&& dim) const -> std::size_t {
    return _sizes[dim];
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension sizes.
  ripple_host_device auto size() const -> std::size_t {
    std::size_t prod_sum = 1;
    unrolled_for<dims>([&] (auto dim) {
      prod_sum *= _sizes[dim];
    });
    return prod_sum;
  }

  //==--- [step] -----------------------------------------------------------==//

  /// Returns the step size to from one element in \p dim to the next element in
  /// \p dim. If the space is used for AOS or SOA storage with \p width elements
  /// in each array, then this will additionally factor in the width.
  /// \param  dim   The dimension to get the step size in.
  /// \param  width The width of the array if the space is for soa or soa.
  /// \tparam Dim   The type of the dimension.
  template <typename Dim>
  ripple_host_device auto step(Dim dim) const -> std::size_t {
    std::size_t res = 1;
    for (auto d : range(static_cast<std::size_t>(dim))) {
      res *= _sizes[d];
    }
    return res;
  }

  //==--- [access] ---------------------------------------------------------==//

  /// Returns a reference to the size of dimension \p dim, which can be used to
  /// set the size of the dimension.
  /// \param[in] dim The dimension size to get a refernece to.
  template <typename Dim>
  ripple_host_device auto operator[](Dim&& dim) -> std::size_t& {
    return _sizes[dim];
  }

 private:
  container_t _sizes; //!< Sizes of the dimensions.
};

} // namespace ripple

#endif // RIPPLE_MULTIDIM_MULTIDIM_SPACE_HPP


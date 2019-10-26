//==--- ripple/container/multidim_storage_info.hpp --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multidim_storage_info.hpp
/// \brief This file imlements a class for multidimensional storage information.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_MULTIDIM_STORAGE_INFO_HPP
#define RIPPLE_CONTAINER_MULTIDIM_STORAGE_INFO_HPP

#include "vec.hpp"

namespace ripple {

/// The DimensionInfo struct defines dimension information over multiple
/// dimensions, specifically the sizes of the dimensions and the steps for each
/// of the dimensions.
/// \tparam Dims The number of dimensions.
template <std::size_t Dimensions>
struct MultidimStorageInfo {
 private:
  /// Defines the number of dimensions
  static constexpr auto dims = Dimensions;

  /// Defines the type of container to store the size and step information.
  using container_t = Vec<std::size_t, dims>;

 public:
  /// Default constructor -- enables creation of empty dimension information.
  MultidimStorageInfo() = default;

  /// Sets the sizes of the dimensions for the storage. This constructor can be
  /// used when the data being stored is not an array type with SOA storage
  /// functionality.
  /// 
  /// \param  sizes The sizes of the dimensions. 
  /// \tparam Sizes The type of the sizes.
  template <typename... Sizes, arithmetic_size_enable_t<dims, Sizes...> = 0>
  ripple_host_device MultidimStorageInfo(Sizes&&... sizes)
  : _sizes{static_cast<std::size_t>(sizes)...} {}

  /// Returns the size of the \p dim dimension.
  /// \param  dim  The dimension to get the size of.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device auto size(Dim&& dim) const -> std::size_t {
    return _sizes[dim];
  }

  /// Returns the step size to from one element in \p dim to the next element in
  /// \p dim. If the elements are stored as SOA, with \p soa_width elements in
  /// each array, then this will additionally factor in the width.
  /// \param  dim       The dimension to get the step size in.
  /// \param  soa_width The width of the array if the storage is soa.  
  /// \tparam Dim       The type of the dimension.
  template <typename Dim>
  ripple_host_device auto step(Dim dim, std::size_t soa_width = 1) const
  -> std::size_t {
    if (dim == 0) { return 1; }

    std::size_t res = soa_width;
    for (auto d : range(static_cast<std::size_t>(dim))) {
      res *= _sizes[d];
    }
    return res;
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension 
  ripple_host_device auto size() const -> std::size_t {
    std::size_t prod_sum = 1;
    unrolled_for<dims>([&] (auto dim) {
      prod_sum *= _sizes[dim];
    });
    return prod_sum;
  }

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

#endif // RIPPLE_CONTAINER_MULTIDIM_STORAGE_INFO_HPP

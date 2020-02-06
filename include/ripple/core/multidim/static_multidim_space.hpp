//==--- ripple/core/multidim/static_multidim_space.hpp ---------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
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

/// The StaticMultidimSpace struct defines spatial information over multiple
/// dimensions, specifically the sizes of the dimensions and the steps required
/// to get from one element to another in the dimensions of the space. The
/// static nature of the space means that it can't be modified, and the size and
/// steps for the space are all known at compile time, which makes using this
/// space more efficient, all be it with less flexibility.
/// \tparam Sizes The sizes of the dimensions of the space.
template <std::size_t... Sizes>
struct StaticMultidimSpace : 
  public MultidimSpace<StaticMultidimSpace<Sizes...>> {
 private:
  /// Defines the number of dimension for the space.
  static constexpr auto dims = sizeof...(Sizes);

  std::size_t _padding = 0; //!< The amount of padding for the space.
  
 public:
  //==--- [construction] ---------------------------------------------------==//
  
  /// Default constructor.
  ripple_host_device constexpr StaticMultidimSpace() = default;

  /// Constructor to set the padding for the space.
  /// \param padding The amount of padding for the space.
  ripple_host_device constexpr StaticMultidimSpace(std::size_t padding)
  : _padding{padding} {}

  //==--- [padding] --------------------------------------------------------==//
  //
  /// Returns the amount of padding for each side of each dimension for the
  /// space.
  ripple_host_device constexpr auto padding() -> std::size_t& {
    return _padding;
  }

  /// Returns the amount of padding for each side of each dimension for the
  /// space.
  ripple_host_device constexpr auto padding() const -> std::size_t {
    return _padding;
  }

  /// Returns the total amount of padding for the dimension, which is twice the
  /// side padding.
  ripple_host_device constexpr auto dim_padding() const -> std::size_t {
    return _padding * 2;
  }

  //==--- [size] -----------------------------------------------------------==//
  
  /// Returns the number of dimensions for the space.
  ripple_host_device constexpr auto dimensions() const -> std::size_t {
    return dims;
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space, including the padding for the space. This is the
  /// product sum of the dimension sizes including padding.
  ripple_host_device constexpr auto size() const -> std::size_t {
    return ((Sizes + dim_padding()) * ... * std::size_t{1});
  }

  /// Returns the size of the dimension with padding.
  /// \param  dim The dimension to get the size for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> std::size_t {
    constexpr std::size_t sizes[dims] = { Sizes... };
    return sizes[dim] + dim_padding();
  }

  /// Returns the size of the \p dim dimension.
  /// \param  dim  The dimension to get the size of.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto internal_size(Dim&& dim) const 
  -> std::size_t {
    constexpr std::size_t sizes[dims] = { Sizes... };
    return sizes[dim];
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space, without the padding.
  ripple_host_device constexpr auto internal_size() const -> std::size_t {
    return (Sizes * ... * std::size_t{1});
  }

  //==--- [step] -----------------------------------------------------------==//

  /// Returns the step size to from one element in \p dim to the next element in
  /// \p dim. 
  /// \param  dim   The dimension to get the step size in.
  /// \tparam Dim   The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto step(Dim&& dim) const -> std::size_t {
    constexpr std::size_t sizes[dims] = { Sizes... };
    std::size_t res = 1;
    for (std::size_t i = 0; i < static_cast<std::size_t>(dim); ++i) {
      res *= sizes[i] + dim_padding();
    }
    return res;
  }
};

} // namespace ripple

#endif // RIPPLE_MULTIDIM_STATIC_MULTIDIM_SPACE_HPP

//==--- ripple/core/multidim/multidim_space.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
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
#include <ripple/core/container/vec.hpp>
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

/// The MultidimSpace defines an interface for classes when define a
/// multidimensional space.
/// \tparam Impl The implementation of the interface.
template <typename Impl>
struct MultidimSpace {
 private:
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
  /// Returns a reference to the amount of padding for each side of each
  /// dimension in the space.
  ripple_host_device constexpr auto padding() -> std::size_t& {
    return impl()->padding();
  }

  /// Returns the amount of padding for each side of each dimension in the
  /// space.
  ripple_host_device constexpr auto padding() const -> std::size_t {
    return impl()->padding();
  }

  /// Returns the total amount of padding for each dimension, which is the sum
  /// of the padding on each side of the dimension.
  ripple_host_device constexpr auto dim_padding() const -> std::size_t {
    return impl()->dim_padding();
  }

  /// Returns the number of dimensions in the space.
  ripple_host_device constexpr auto dimensions() const -> std::size_t {
    return SpaceTraits<Impl>::dimensions;
  }

  /// Returns the size of the \p dim dimension, including the padding.
  /// \param  dim  The dimension to get the size of.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> std::size_t {
    return impl()->size(std::forward<Dim>(dim));
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space, including the padding for each of the dimensions.
  ripple_host_device constexpr auto size() const -> std::size_t {
    return impl()->size();
  }

  /// Returns the size of the \p dim dimension, without padding.
  /// \param  dim  The dimension to get the size oAf.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto internal_size(Dim&& dim) const 
  -> std::size_t {
    return impl()->internal_size(std::forward<Dim>(dim));
  }

  /// Returns the total internal size of the N dimensional space i.e the total
  /// number of elements in the space without padding.A
  ripple_host_device constexpr auto internal_size() const -> std::size_t {
    return impl()->internal_size();
  }

  /// Returns the step size to from one element in \p dim to the next element in
  /// \p dim.
  /// \param  dim   The dimension to get the step size in.
  /// \tparam Dim   The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto step(Dim&& dim) const -> std::size_t {
    return impl()->step(std::forward<Dim>(dim));
  }
};

} // namespace ripple

#endif // RIPPLE_MULTIDIM_MULTIDIM_SPACE_HPP


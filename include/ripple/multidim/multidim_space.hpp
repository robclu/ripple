//==--- ripple/multidim/multidim_space.hpp ----------------- -*- C++ -*- ---==//
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

#include <ripple/container/vec.hpp>
#include <ripple/utility/type_traits.hpp>

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
  ripple_host_device auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:

  /// Returns the amount of padding in dimension \p dim for the space. The 
  /// padding is the number of elements at the beginning and end of the space,
  /// which are not used for computation. This must returns the padding at one
  /// side of the dimension in the space, which is the same on the other size of
  /// the dimension for the space.
  /// \param[in] dim The dimension to get the padding for.
  template <typename Dim>
  ripple_host_device constexpr auto padding(Dim&& dim) -> std::size_t {
    return impl()->padding(std::forward<Dim>(dim));
  }

  /// Returns the size of the \p dim dimension.
  /// \param  dim  The dimension to get the size of.
  /// \tparam Dim  The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> std::size_t {
    return impl()->size(std::forward<Dim>(dim));
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension sizes.
  ripple_host_device constexpr auto size() const -> std::size_t {
    return impl()->size();
  }

  /// Returns the step size to from one element in \p dim to the next element in
  /// \p dim. If the space is used for AOS or SOA storage with \p width elements
  /// in each array, then this will additionally factor in the width.
  /// \param  dim   The dimension to get the step size in.
  /// \param  width The width of the array if the space is for soa or soa.
  /// \tparam Dim   The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto step(Dim&& dim, std::size_t width = 1) const
  -> std::size_t {
    return impl()->step(std::forward<Dim>(dim), width);
  }
};

} // namespace ripple

#endif // RIPPLE_MULTIDIM_MULTIDIM_SPACE_HPP


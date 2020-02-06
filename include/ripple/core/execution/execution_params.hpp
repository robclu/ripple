//==--- ripple/core/execution/execution_params.hpp -------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  execution_params.hpp
/// \brief This file defines an interface for execution parameters.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP
#define RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP

#include "execution_traits.hpp"
#include <ripple/core/utility/dim.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/// The ExecParams struct defines an interface for all execution parameter
/// implementations. The purpose of execution paramters are to customize the
/// execution space and to create data and iterators over the data in the
/// executin space.
/// \tparam Impl The implementation of the interface.
template <typename Impl>
struct ExecParams {
 private:
  /// Defines the type of the implementation.
  using impl_t   = std::decay_t<Impl>;
  /// Defines the traits for the implementation.
  using traits_t = ExecTraits<impl_t>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }
 public:
  //==--- [size] -----------------------------------------------------------==//
  
  /// Returns the total size of the execution space, for Dims dimensions.
  /// \tparam Dims The number of dimensions to get the size for.
  template <std::size_t Dims>
  ripple_host_device constexpr auto size() const -> std::size_t {
    return impl()->template size<Dims>();
  }

  /// Returns the size of the space in the \p dim dimension.
  /// \tparam dim The dimension to get the size of the space for.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> std::size_t {
    return impl()->size(std::forward<Dim>(dim));
  }

  //==--- [properties] -----------------------------------------------------==//

  /// Returns the amount of padding for the execution space.
  ripple_host_device constexpr auto padding() const -> std::size_t {
    return impl()->padding();
  }

  //==--- [creation] -------------------------------------------------------==//

  /// Returns an iterator over a memory space pointed to by \p data, for a
  /// number of dimensions.
  /// \param  data A pointer to the memory space data.
  /// \tparam Dims The number of dimensions for the iterator.
  /// \tparam T    The type of the data to create the iterator in.  
  template <size_t Dims, typename T>
  ripple_host_device auto iterator(T* data) const {
    return impl()->template iterator<Dims>(data);
  }

  /// Returns the number of bytes required to allocator data for the space.
  /// \tparam Dims The number of dimensions to allocate for.
  template <std::size_t Dims>
  ripple_host_device constexpr auto allocation_size() const -> std::size_t {
    return impl()->template allocation_size<Dims>();
  }
};

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP

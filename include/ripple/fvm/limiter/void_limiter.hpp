//==--- ripple/fvm/limiting/void_limiter.hpp --------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  void_limiter.hpp
/// \brief This file defines an implementation of a void limiter.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_LIMITER_VOID_LIMITER_HPP
#define RIPPLE_LIMITER_VOID_LIMITER_HPP

#include "limiter.hpp"

namespace ripple::fv {

/// The VoidLimiter class implements the Limiter interface, but performs no
/// limiting, which can be used to reduce higher order schemes to lower order
/// schemes. 
class VoidLimiter : public Limiter<VanLeerLimiter> {
  /// Defines the type of the base limiter.
  using base_limiter_t = Limiter<VanLeerLimiter>;

 public:
  /// Implementation of the limit function which applies the limiting to an
  /// iterator. This does not perform any limiting, ans imply returns zero.
  /// \param  state_it  The state iterator to limit.
  /// \param  dim       The dimension to limit in.
  /// \tparam Iterator  The type of the state iterator.
  /// \tparam Dim       The dimension to limit in.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto limit_impl(Iterator&& state_it, Dim&& dim) 
  const -> typename state_traits_t<decltype(*state_it)>::contiguous_layout_t {
    using return_t = 
      typename state_traits_t<decltype(*state_it)>::contiguous_layout_t;
    return return_t{0};
  }
};

} // namespace ripple::fv

#endif // RIPPLE_LIMITER_VOID_LIMITER_HPP
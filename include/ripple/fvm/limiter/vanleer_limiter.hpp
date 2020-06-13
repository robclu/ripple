//==--- ripple/fvm/limiting/vanleer_limiter.hpp ------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vanleer_limiter.hpp
/// \brief This file defines an implementation of the Van Leer limiting.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_LIMITER_VANLEER_LIMITER_HPP
#define RIPPLE_LIMITER_VANLEER_LIMITER_HPP

#include "limiter.hpp"
#include <ripple/core/math/math.hpp>

namespace ripple::fv {

/// Implemenation of the limtier interface which performs VanLeer limiting.
class VanLeerLimiter : public Limiter<VanLeerLimiter> {
  /// Defines the type of the base limiter.
  using base_limiter_t = Limiter<VanLeerLimiter>;

 public:
  /// Implementation of the limit function which applies the limiting to an
  /// iterator, calling the limit method on each of the iterator elements.
  /// \param  state_it  The state iterator to limit.
  /// \param  dim       The dimension to limit in.
  /// \tparam Iterator  The type of the state iterator.
  /// \tparam Dim       The dimension to limit in.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto limit_impl(Iterator&& state_it, Dim&& dim) 
  const -> typename state_traits_t<decltype(*state_it)>::contiguous_layout_t {
    return base_limiter_t::limit_generic(
      std::forward<Iterator>(state_it), std::forward<Dim>(dim));
  }

  /// Returns the limited value of a single element, using the van leer limiting
  /// method.
  /// \param left The left element to limit on.
  /// \param right The right element to limit on.
  template <typename T>
  ripple_host_device constexpr auto limit_single(
    const T& left, const T& right
  ) const -> T {
    using value_t        = std::decay_t<T>;
    constexpr auto zero  = value_t{0}, one = value_t{1}, two = value_t{2};
    const     auto ratio = left / right;
    return (ratio <= zero || right == zero) 
      ? zero 
      : two * std::min(ratio, one) / (one + ratio);
  }
};

} // namespace ripple::fv

#endif // RIPPLE_LIMITER_VANLEER_LIMITER_HPP
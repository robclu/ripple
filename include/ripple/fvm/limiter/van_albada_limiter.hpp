//==--- ripple/fvm/limiting/vanalbada_limiter.hpp ---------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vanalbada_limiter.hpp
/// \brief This file defines an implementation of the Van Albada limiting.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_LIMITER_VANALBADA_LIMITER_HPP
#define RIPPLE_LIMITER_VANALBADA_LIMITER_HPP

#include "limiter.hpp"
#include <ripple/core/math/math.hpp>

namespace ripple::fv {

/// Implemenation of the limtier interface which performs VanAlbada limiting.
class VanAlbadaLimiter : public Limiter<VanAlbadaLimiter> {
  /// Defines the type of the base limiter.
  using base_limiter_t = Limiter<VanAlbadaLimiter>;

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

  /// Returns the limited value of a single element, using the van Albada
  /// limiting method.
  /// \param left The left element to limit on.
  /// \param right The right element to limit on.
  template <typename T>
  ripple_host_device constexpr auto limit_single(
    const T& left, const T& right
  ) const -> T {
    using value_t       = std::decay_t<T>;
    constexpr auto zero = value_t{0}, one = value_t{1}, two = value_t{2};
    const     auto r    = left / right;
    const     auto er   = two / (one + r);
    return (r <= zero || right == zero) 
      ? zero 
      : std::min(r * (one + r) / (one + r * r), er);
  }
};

} // namespace ripple::fv

#endif // RIPPLE_LIMITER_VANALBADA_LIMITER_HPP
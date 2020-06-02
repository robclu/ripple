//==--- ripple/fvm/limiter/limiter.hpp -----------------------*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  limiter.hpp
/// \brief This file defines the interface to which all limiters must conform,
///        and also provides a default implementation which derived limters can
///        invoke if their implementation is not specific.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FVM_LIMITER_LIMITER_HPP
#define RIPPLE_FVM_LIMITER_LIMITER_HPP

//#include <fluidity/algorithm/unrolled_for.hpp>
#include <ripple/core/utility/portability.hpp>
#include <ripple/fvm/state/state_traits.hpp>

namespace ripple::fv {

/// The Limiter class defines the interface to which all limiters
/// must conform. The implementation is provided by the template type.
/// \tparam LimiterImpl The type of the limiter implementation.
template <typename LimiterImpl>
class Limiter {
  /// Defines the type of the reconstructor implementation.
  using impl_t = LimiterImpl;

  /// Returns a pointer to the implementation.
  ripple_host_device auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  ripple_host_device auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

 protected:
  /// Default implementation of the limiting, which requires that the
  /// implementation type has a `limit_single(left, right)` implementation.
  /// \param  state_it The state iterator to limit.
  /// \tparam Iterator The type of the state iterator.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto limit_generic(
    Iterator&& state_it, Dim&& dim
  ) const -> typename state_traits_t<decltype(*state_it)>::contiguous_layout_t {
    using state_t = std::decay_t<decltype(*state_it)>;
    using value_t = typename state_traits_t<state_t>::value_t;
    
    const auto fwrd_diff = state_it.forward_diff(std::forward<Dim>(dim));
    const auto back_diff = state_it.backward_diff(std::forward<Dim>(dim));
    auto limited = value_t{0.5} * (fwrd_diff + back_diff);
                
    unrolled_for<state_traits_t<state_t>::elements>([&] (auto i) {
      limited[i] *= impl()->limit_single(back_diff[i], fwrd_diff[i]);
    });
    return limited;
  }

 public:

  /// Returns the slope in the \p dim dimension after limiting. In the general
  /// case, this will return:
  ///
  /// \begin{equation}
  ///   \eita_i * \frac{1}{2} [\Delta_{i-1} + \Delta_{i+1}]
  /// \end{equation}
  ///
  /// where $\eita_i$ is the limiter of the $ith$ element in the \p state_it
  /// vector, and $\Delta$ is the gradient across the backward or forward face.
  ///
  /// \param   state_it  The state iterator to limit.
  /// \param   dim       The dimension to get the slope in.
  /// \tparam  Iterator  The type of the state iterator.
  /// \tparam  Value     The value which defines the dimension to limit in.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto slope(Iterator&& state_it, Dim&& dim)
  const -> typename state_traits_t<decltype(*state_it)>::contiguous_layout_t {
    return impl()->limit_impl(
      std::forward<Iterator>(state_it), std::forward<Dim>(dim));
  }
};

} // namespace ripple::fv

#endif // RIPPLE_FVM_LIMITER_LIMITER_HPP
//==--- ripple/fv/flux/force.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  force.hpp
/// \brief This file defines an implementation of the Force flux.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FV_FLUX_FORCE_HPP
#define RIPPLE_FV_FLUX_FORCE_HPP

#include "lax_friedrichs.hpp"
#include "richtmyer.hpp"

namespace ripple::fv {

/// The Force type implements the Flux interface to compute the flux
/// between a left and a right state for a given dimension, using the FORCE
/// method, which is:
///
/// \begin{equation}
///   F_{i + \frac{1}{2}} = \frac{1}{2} ( F_{LF} + F_{RM} )
/// \end{equation}
///
/// where ``F_LF`` is the Lax-Friedrichs flux, and ``F_RM`` is the Richtmyer
/// flux.
struct Force : public Flux<Force> {
  /// Computes the flux between the \p l left and \p r right states, in the
  /// dimension \p dim, using the \p eos equation of state, and the \p dt time
  /// resolution and \p dh spatial resolution, returning a new state as the 
  /// result.
  ///
  /// This will fail at compile time if either StateImplL or StateImplR are not
  /// implementations of the State interface or if Eos is not an implementation
  /// of the Eos interface.
  ///
  /// \param  l          The left state for the flux computation.
  /// \param  r          The right state for the flux computation.
  /// \param  eos        The equation of state for the flux computation.
  /// \param  dt         The time resolution for the computation.
  /// \param  dh         The spatial resolution for the computation.
  /// \tparam StateImplL The implementation of the left state interface.
  /// \tparam StateImplR The implementation of the right state interface.
  /// \tparam EosImpl    The type of the equation of state implementation.
  /// \tparam Dim        The type of the dimension specifier.
  /// \tparam T          The data type for the resolutions.
  template <
    typename StateImplL,
    typename StateImplR,
    typename EosImpl   , 
    typename Dim       ,
    typename T
  >
  ripple_host_device auto evaluate(
    const StateImplL& l ,
    const StateImplR& r ,
    const EosImpl&    eos,
    Dim&&             dim,
    T                 dt ,
    T                 dh
  ) const -> typename state_traits_t<StateImplL>::contiguous_layout_t {
    ensure_correct_interfaces<StateImplL, StateImplR, EosImpl>();
    return (
      LaxFriedrichs().evaluate(l, r, eos, std::forward<Dim>(dim), dt, dh) +
      Richtmyer().evaluate(l, r, eos, std::forward<Dim>(dim), dt, dh)
    ) * T{0.5};
  }

  /// Computes the flux between the \p l left and \p r right states, using the
  /// \p eos equation of state, and the \p dt time resolution and \p dh spatial
  /// resolution. This stores the result in the \p res state.
  ///
  /// This will fail at compile time if either StateImplL, StateImplR or
  /// ResultImpl are not implementations of the State interface or if Eos is 
  /// not an implementation of the Eos interface.
  ///
  /// \param  l          The left state for the flux computation.
  /// \param  r          The right state for the flux computation.
  /// \param  res        The state to store the result in.
  /// \param  eos        The equation of state for the flux computation.
  /// \param  dt         The time resolution for the computation.
  /// \param  dh         The spatial resolution for the computation.
  /// \tparam StateImplL The implementation of the left state interface.
  /// \tparam StateImplR The implementation of the right state interface.
  /// \tparam ResultImpl The implementation type  of the result state.
  /// \tparam EosImpl    The type of the equation of state implementation.
  /// \tparam Dim        The type of the dimension specifier.
  /// \tparam T          The data type for the resolutions.
  template <
    typename StateImplL,
    typename StateImplR,
    typename ResultImpl,
    typename EosImpl   , 
    typename Dim       ,
    typename T
  >
  ripple_host_device auto evaluate(
    const StateImplL& l     ,
    const StateImplR& r     ,
    ResultImpl&       result,
    const EosImpl&    eos   ,
    Dim&&             dim   ,
    T                 dt    ,
    T                 dh
  ) const -> void {
    ensure_correct_interfaces<StateImplL, StateImplR, EosImpl>();
    static_assert(is_state_v<ResultImpl>,
      "Flux requires arg result to implement the State intreface."
    );

    using value_t = typename state_traits_t<ResultImpl>::value_t;
    LaxFriedrichs().evaluate(l, r, result, eos, dim, dt, dh);
    result += Richtmyer().evaluate(l, r, eos, dim, dt, dh);
    result *= value_t{0.5};
  }
};

} // namespace ripple::fv

#endif // RIPPLE_FVM_FLUX_FORCE__HPP

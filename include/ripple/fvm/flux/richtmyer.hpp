//==--- ripple/fv/flux/richtmyer.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  richtmyer.hpp
/// \brief This file defines an implementation of the Richtmyer flux.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FV_FLUX_RICHTMYER_HPP
#define RIPPLE_FV_FLUX_RICHTMYER_HPP

#include "flux.hpp"
#include <ripple/fvm/eos/eos.hpp>
#include <ripple/fvm/state/state_traits.hpp>

namespace ripple::fv {

/// The Richtmyer type implements the Flux interface to compute the flux
/// between a left and a right state for a given dimension.
///
/// For more information, see equation 5.79 in Toro - Riemann Solvers and
/// Numerical Method for Fluid Dynamics.
///
/// The RM flux is computed as follows:
///
/// \begin{equation}
///   F_{i+ \frac{1}{2}} = 
///     F \left( U_{i + \frac{1}{2}}^{n + \frac{1}{2}} \right)
/// \end{equation}
///
/// where 
///
/// \begin{equation}
///   U_{i + \frac{1}{2}}^{n + \frac{1}{2}} =
///     \frac{1}{2} \left( U_i^n + U_{i+ 1}^n \right) +
///     \frac{1}{2} \frac{\delta t}{\deta h}
///       \left( F_i^n - F_{i + 1}^n \right)
/// \end{equation}
struct Richtmyer : public Flux<Richtmyer> {
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
    using res_t = typename state_traits_t<StateImplL>::contiguous_layout_t;
    return res_t( 
      T{0.5} * (l + r + (dt / dh) * (l.flux(eos, dim) - r.flux(eos, dim)))
    ).flux(eos, dim);
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

    using res_t   = ResultImpl;
    using value_t = typename state_traits_t<StateImplL>::value_t;
    constexpr auto _0_5 = value_t{0.5};
    auto rm_state = res_t{ 
      _0_5 * (l + r + (dt / dh) * (l.flux(eos, dim) - r.flux(eos, dim)))
    };
    result = rm_state.flux(eos, dim);
  }
};

} // namespace ripple::fv

#endif // RIPPLE_FVM_FLUX_RICHTMYER_HPP


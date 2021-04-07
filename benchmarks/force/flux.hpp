/**=--- ripple/benchmarks/flux.hpp ------------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  flux.hpp
 * \brief This file defines an implementation of the Lax-Friedrichs flux.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARKS_FLUX_FLUX_HPP
#define RIPPLE_BENCHMARKS_FLUX_FLUX_HPP

#include <ripple/iterator/iterator_traits.hpp>

/**
 * The LaxFriedrichs type computes the flux between a left and a right state
 * for a given dimension.
 *
 * For more information, see equation 5.77 in Toro - Riemann Solvers and
 * Numerical Method for Fluid Dynamics.
 *
 * The LF flux is computed as follows:
 *
 * \begin{equation}
 *   F_{i + \frac{1}{2}} =
 *     \frac{1}{2} \left( F_{i}^n + F_{i + 1}^n \right) +
 *     \frac{1}{2} \frac{\delta x}{\delta t}
 *       \left( U_{i}^n - U_{i + 1}^n \right)
 * \end{equation}
 */
struct LaxFriedrichs {
  /**
   * Computes the flux between the \p l left and \p r right states, in the
   * dimension \p dim, using the \p eos equation of state, and the \p dt time
   * resolution and \p dh spatial resolution, returning a new state as the
   * result.
   *
   * This will fail at compile time if either StateImplL or StateImplR are not
   * implementations of the State interface or if Eos is not an implementation
   * of the Eos interface.
   *
   * \param  l          The left state for the flux computation.
   * \param  r          The right state for the flux computation.
   * \param  eos        The equation of state for the flux computation.
   * \param  dt         The time resolution for the computation.
   * \param  dh         The spatial resolution for the computation.
   * \tparam StateImplL The implementation of the left state interface.
   * \tparam StateImplR The implementation of the right state interface.
   * \tparam EosImpl    The type of the equation of state implementation.
   * \tparam Dim        The type of the dimension specifier.
   * \tparam T          The data type for the resolutions.
   */
  template <
    typename StateImplL,
    typename StateImplR,
    typename EosImpl,
    typename Dim,
    typename T,
    ripple::non_iterator_enable_t<StateImplL> = 0>
  ripple_host_device auto operator()(
    const StateImplL& l,
    const StateImplR& r,
    const EosImpl&    eos,
    Dim&&             dim,
    T                 sc) const noexcept -> typename StateImplL::ContigState {
    return (l.flux(eos, ripple_forward(dim)) +
            r.flux(eos, ripple_forward(dim)) + ((l - r) * sc)) *
           T{0.5};
  }
};

/**
 * The Richtmyer type computes the flux between a left and a right state for
 * a given dimension.
 *
 * For more information, see equation 5.79 in Toro - Riemann Solvers and
 * Numerical Method for Fluid Dynamics.
 *
 * The RM flux is computed as follows:
 *
 * \begin{equation}
 *   F_{i+ \frac{1}{2}} =
 *     F \left( U_{i + \frac{1}{2}}^{n + \frac{1}{2}} \right)
 * \end{equation}
 *
 * where
 *
 * \begin{equation}
 *   U_{i + \frac{1}{2}}^{n + \frac{1}{2}} =
 *     \frac{1}{2} \left( U_i^n + U_{i+ 1}^n \right) +
 *     \frac{1}{2} \frac{\delta t}{\deta h}
 *       \left( F_i^n - F_{i + 1}^n \right)
 * \end{equation}
 */
struct Richtmyer {
  /**
   * Computes the flux between the \p l left and \p r right states, in the
   * dimension \p dim, using the \p eos equation of state, and the \p dt time
   * resolution and \p dh spatial resolution, returning a new state as the
   * result.
   *
   *
   * \param  l          The left state for the flux computation.
   * \param  r          The right state for the flux computation.
   * \param  eos        The equation of state for the flux computation.
   * \param  dt         The time resolution for the computation.
   * \param  dh         The spatial resolution for the computation.
   * \tparam StateImplL The implementation of the left state interface.
   * \tparam StateImplR The implementation of the right state interface.
   * \tparam EosImpl    The type of the equation of state implementation.
   * \tparam Dim        The type of the dimension specifier.
   * \tparam T          The data type for the resolutions.
   */
  template <
    typename StateImplL,
    typename StateImplR,
    typename EosImpl,
    typename Dim,
    typename T,
    ripple::non_iterator_enable_t<StateImplL> = 0>
  ripple_host_device auto operator()(
    const StateImplL& l,
    const StateImplR& r,
    const EosImpl&    eos,
    Dim&&             dim,
    T                 sc) const noexcept -> typename StateImplL::ContigState {
    using ResultType = typename StateImplL::ContigState;
    return ResultType{
      T{0.5} * (l + r +
                sc * (l.flux(eos, ripple_forward(dim)) -
                      r.flux(eos, ripple_forward(dim))))}
      .flux(eos, ripple_forward(dim));
  }
};

/**
 * The Force type computes the flux between a left and a right state for a
 * given dimension, using the FORCE method, which is:
 *
 * \begin{equation}
 *   F_{i + \frac{1}{2}} = \frac{1}{2} ( F_{LF} + F_{RM} )
 * \end{equation}
 *
 * where ``F_LF`` is the Lax-Friedrichs flux, and ``F_RM`` is the Richtmyer
 * flux.
 */
struct Force {
  /**
   * Computes the flux between the \p l left and \p r right states, in the
   * dimension \p dim, using the \p eos equation of state, and the \p dt time
   * resolution and \p dh spatial resolution, returning a new state as the
   * result.
   *
   *
   * \param  l          The left state for the flux computation.
   * \param  r          The right state for the flux computation.
   * \param  eos        The equation of state for the flux computation.
   * \param  dt         The time resolution for the computation.
   * \param  dh         The spatial resolution for the computation.
   * \tparam StateImplL The implementation of the left state interface.
   * \tparam StateImplR The implementation of the right state interface.
   * \tparam EosImpl    The type of the equation of state implementation.
   * \tparam Dim        The type of the dimension specifier.
   * \tparam T          The data type for the resolutions.
   */
  template <
    typename StateImplL,
    typename StateImplR,
    typename EosImpl,
    typename Dim,
    typename T>
  ripple_host_device auto operator()(
    const StateImplL& l,
    const StateImplR& r,
    const EosImpl&    eos,
    Dim&&             dim,
    T                 sc) const noexcept -> typename StateImplL::ContigState {
    constexpr auto lf = LaxFriedrichs();
    constexpr auto rm = Richtmyer();
    return T{0.5} * (lf(l, r, eos, ripple_forward(dim), sc) +
                     rm(l, r, eos, ripple_forward(dim), sc));
  }
};

#endif // RIPPLE_BENCHMARKS_FLUX_FLUX_HPP

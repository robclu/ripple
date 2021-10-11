/**=--- ripple/benchmarks/force/flux_kokkos.hpp ------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  flux_kokkos.hpp
 * \brief This file defines an implementation of the Lax-Friedrichs flux.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARKS_FLUX_FLUX_KOKKOS_HPP
#define RIPPLE_BENCHMARKS_FLUX_FLUX_KOKKOS_HPP

#include <ripple/utility/portability.hpp>
#include "state_kokkos.hpp"

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
  template <typename E, typename T, size_t D>
  ripple_all auto operator()(
    const State<T, D>& l, const State<T, D>& r, const E& eos, size_t dim, T sc)
    const noexcept -> State<T, D> {
    return (l.flux(eos, dim) + r.flux(eos, dim) + ((l - r) * sc)) * T{0.5};
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
  template <typename E, typename T, size_t D>
  ripple_all auto operator()(
    const State<T, D>& l, const State<T, D>& r, const E& eos, size_t dim, T sc)
    const noexcept -> State<T, D> {
    const auto temp =
      (l + r + (l.flux(eos, dim) - r.flux(eos, dim)) * sc) * T{0.5};
    return temp.flux(eos, dim);
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
  template <typename E, typename T, size_t D>
  ripple_all auto operator()(
    const State<T, D>& l, const State<T, D>& r, const E& eos, size_t dim, T sc)
    const noexcept -> State<T, D> {
    constexpr auto lf = LaxFriedrichs();
    constexpr auto rm = Richtmyer();
    return (lf(l, r, eos, dim, sc) + rm(l, r, eos, dim, sc)) * T{0.5};
  }
};

#endif // RIPPLE_BENCHMARKS_FLUX_FLUX_KOKKOS_HPP
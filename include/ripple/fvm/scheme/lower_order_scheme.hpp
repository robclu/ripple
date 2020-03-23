//==--- ripple/fvm/scheme/lower_order_scheme.hpp ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  lower_order_scheme.hpp
/// \brief This file defines an implementation of the scheme interface to
///        compute the flux between two states using a lower order method.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_SCHEME_LOWER_ORDER_SCHEME_HPP
#define RIPPLE_SCHEME_LOWER_ORDER_SCHEME_HPP

#include "scheme.hpp"
#include <ripple/fvm/state/state_traits.hpp>

namespace ripple::fv {

/// The LowerOrderScheme type implements the Scheme interface to compute the
/// intercell flux between two states. Since this is a lower order method, it
/// does not perform any reconstruction of the states, and simply uses the flux
/// method to compute the intercell flux.
struct LowerOrderScheme : public Scheme<LowerOrderScheme> {
  /// Computes the flux delta between the faces of the cell to which the \p it
  /// iterator points to, in the dimension given by \p dim. For the x dimenison,
  /// this computes 
  ///
  /// \begin{equation}
  ///   F_{i - \frac{1}{2}} - F_{i + \frac{1}{2}}
  /// \end{equation}
  ///
  /// and similarly for the other dimensions. It used the provided
  /// implementation of the Flux interface to evaluate the flux at the face, i.e
  /// 
  /// \begin{equation}
  ///   F_{i - \frac{1}{2}}, F_{i - \frac{1}{2}}
  /// \end{equation}.
  ///
  /// If either the \p eos or the \p flux do not implement the Eos and Flux
  /// interfaces, respectively, this will cause a compile time error.
  ///
  /// Additionally, if the \p i is not an iterator, then this will fail.
  ///
  /// \param  it       An iterator to the data.
  /// \param  eos      The equation of state for the flux computation.
  /// \param  flux     The type to compute the flux/RP between two states.
  /// \param  dt       The time resolution for the computation.
  /// \param  dh       The spatial resolution for the computation.
  /// \tparam Iterator The type of the iterator.
  /// \tparam EosImpl  The type of the equation of state implementation.
  /// \tparam FluxImpl The type of the flux computer.
  /// \tparam Dim      The type of the dimension specifier.
  /// \tparam T        The data type for the resolutions.
  template <
    typename Iterator,
    typename EosImpl , 
    typename FluxImpl,
    typename Dim     ,
    typename T
  >
  ripple_host_device auto flux_delta(
    Iterator&&     it  ,
    const EosImpl& eos ,
    FluxImpl&&     flux,
    Dim&&          dim ,
    T              dt  ,
    T              dh
  ) const -> typename state_traits_t<decltype(*it)>::contiguous_layout_t {
    ensure_correct_interfaces<Iterator, EosImpl, FluxImpl>();
    return flux.evaluate(*it.offset(dim, -1), *it, eos, dim, dt, dh) -
      flux.evaluate(*it, *it.offset(dim, 1), eos, dim, dt, dh);
  }
};

} // namespace ripple::fv

#endif // RIPPLE_SCHEME_LOWER_ORDER_SCHEME_HPP

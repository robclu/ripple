//==--- ripple/fvm/scheme/scheme.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  scheme.hpp
/// \brief This file defines a static interface for a computational scheme,
///        which is a means for computing the intercell flux between two states.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_SCHEME_SCHEME_HPP
#define RIPPLE_SCHEME_SCHEME_HPP

#include <ripple/fvm/eos/eos_traits.hpp>
#include <ripple/fvm/flux/flux.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>
#include <ripple/core/utility/portability.hpp>
#include <ripple/core/utility/type_traits.hpp>

namespace ripple::fv {

/// The Scheme type defines an interface for computing the intercell flux
/// difference between the faces of a cell in a given dimension.
/// \tparam Impl The implementation of the scheme interface.
template <typename Impl>
class Scheme {
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
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
  /// \parma  it       The iterator which points to the cell data.
  /// \param  eos      The equation of state for the data.
  /// \param  flux     The type to compute the flux/RP between states.
  /// \param  dt       The time resolution for the computation.
  /// \param  dh       The spatial resolution for the computation.
  /// \tparam Iterator The type of the iterator.
  /// \tparam EosImpl  The type of the equation of state implementation.
  /// \tparam FluxImpl The implementation of the flux interface.
  /// \tparam Dim      The type of the dimension specifier.
  /// \tparam T        The data type for the resolutions.
  template <
    typename Iterator  ,
    typename EosImpl   , 
    typename FluxImpl  ,
    typename Dim       ,
    typename T
  >
  ripple_host_device auto flux_delta(
    Iterator&& it, const EosImpl& eos, FluxImpl&& flux, Dim&& dim, T dt, T dh
  ) const -> typename state_traits_t<decltype(*it)>::contiguous_layout_t {
    ensure_correct_interfaces();
    return impl()->flux_deta(
      std::forward<Iterator>(it)  ,
      eos                         ,
      std::forward<FluxImpl>(flux),
      std::forward<Dim>(dim)      ,
      dt                          , 
      dh
    );
  }

 private:
  /// Ensures that the types meet the interface requirements.
  /// \tparam Iterator The type of the iterator.
  /// \tparam EosImpl  The type of the equation of state implementation.
  /// \tparam FluxImpl The type of the flux implementation.
  template <typename Iterator, typename EosImpl, typename FluxImpl>
  ripple_host_device constexpr auto ensure_correct_interfaces() const -> void {
    static_assert(is_iterator_v<Iterator>, 
      "Iterator, it, must be an iterator."
    );
    static_assert(is_eos_v<EosImpl>,
      "Equation of state, eos, must implement the Eos interface."
    );
    static_assert(is_flux_v<FluxImpl>,
      "Flux evaluator, flux, must implement the Flux interface."
    );
  }
};

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T implements the Scheme interface.
/// \tparam T The type to determine if implements the Scheme interface.
template <typename T>
static constexpr auto is_scheme_v = 
  std::is_base_of_v<Scheme<std::decay_t<T>>, std::decay_t<T>>;

} // namespace ripple::fv

#endif // RIPPLE_SCHEME_SCHEME_HPP

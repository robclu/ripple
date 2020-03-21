//==--- ripple/fvm/flux/flux.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux.hpp
/// \brief This file defines a static interface for computing the flux.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FLUX_FLUX_HPP
#define RIPPLE_FLUX_FLUX_HPP

#include <ripple/fvm/eos/eos.hpp>
#include <ripple/fvm/state/state.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple::fv {

/// The Flux class defines an interface for computing the flux between two
/// states in a given dimension.
/// \tparam Impl The implementation of the interface.
template <typename Impl>
class Flux {
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
  ) const -> typename state_traits_t<StateImplL>::contiguous_state_t {
    ensure_correct_interfaces<StateImplL, StateImplR, EosImpl>();
    return impl()->evaluate(l, r, eos, std::forward<Dim>(dim), dt, dh);
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
    return impl()->evaluate(l, r, result, eos, std::forward<Dim>(dim), dt, dh);
  }

 protected:
  /// Ensures that StateImplL and StateImplR and implementations of the State
  /// interface, and that EosImpl is an implementation of the Eos interface.
  /// \tparam StateImplL The implementation type of the left state.
  /// \tparam StateImplR The implementation type of right state.
  /// \tparam EosImpl    The implementation type of the equation of state.
  template <typename StateImplL, typename StateImplR, typename EosImpl>
  ripple_host_device constexpr auto ensure_correct_interfaces() const -> void {
    static_assert(is_state_v<StateImplL>,
      "Flux requires arg l to implement the State interface."
    );
    static_assert(is_state_v<StateImplR>,
      "Flux requires arg r to implement the State interface."
    );
    static_assert(is_eos_v<EosImpl>,
      "Flux required arg eos to implement the Eos interface."
    );
  }
};

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T implements the Flux interface, otherwise return
/// false.
/// \tparam T The type to determine if implements the flux interface.
template <typename T>
static constexpr auto is_flux_v = 
  std::is_base_of_v<Flux<std::decay_t<T>>, std::decay_t<T>>;

} // namespace ripple::fv

#endif // RIPPLE_FVM_FLUX_FLUX_HPP

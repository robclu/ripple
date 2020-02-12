//==--- ripple/fvm/eos/eos.hpp ----------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  eos.hpp
/// \brief This file defines a static interface for an equation of state.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EOS_EOS_HPP
#define RIPPLE_EOS_EOS_HPP

#include "eos_traits.hpp"

namespace ripple::fv {

/// The Eos class defines a state interface for equations of state.
/// \tparam Impl The implementation of the iterface.
template <typename Impl>
class Eos {
  /// Defines the type of the implementation.
  using impl_t   = std::decay_t<Impl>;
  /// Defines the traits for the equation of state.
  using traits_t = EosTraits<impl_t>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
  /// Defines the data type used by the equation of state.
  using value_t = typename traits_t::value_t;

  /// Returns the value of the adiabatic index for the equation of state.
  ripple_host_device constexpr auto adi() -> value_t& {
    return impl()->adi();
  }

  /// Returns the value of the adiabatic index for the equation of state.
  ripple_host_device constexpr auto adi() const -> value_t {
    return impl()->adi();
  }
  
  /// Evaluates the equation of state for the given \p state.
  /// \param  state The state to use to evaluate the equation of state.
  /// \tparam State The type of the state.
  template <typename State>
  ripple_host_device constexpr auto eos(const State& state) const -> value_t {
    return impl()->eos(state);
  }

  /// Calculates the speed of sound for the equation of state for the given \p
  /// state.
  /// \param  state The state to use to compute the sound speed.
  /// \tparam State The type of the state.
  template <typename State>
  ripple_host_device constexpr auto sound_speed(const State& state) const 
  -> value_t {
    return impl()->sound_speed(state);
  } 
};

} // namespace ripple::fv

#endif // RIPPLE_EOS_EOS_HPP


//==--- ripple/fvm/eos/ideal_gas.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  ideal_gas.hpp
/// \brief This file implements an ideal gas class which is an implementation
///        of the equation of state interface for an ideal gas.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EOS_IDEAL_GAS_HPP
#define RIPPLE_EOS_IDEAL_GAS_HPP

#include "eos.hpp"
#include <cmath>

namespace ripple::fv {

/// The IdealGas type implements the equation of state interface which
/// represents an ideal gas.
/// \tparam T The type of the data for the ideal gas.
template <typename T>
struct IdealGas : public Eos<IdealGas<T>> {
 private:
  /// Defines a constant value of one which is used throughout the
  /// implementation.
  static constexpr auto _1 = T{1};

 public:
  /// Defines the type of the data used by the material.
  using value_t = std::decay_t<T>;

  /// Sets the value of the gas to have the default adiabatic index of 1.4.
  constexpr IdealGas() = default;

  /// Sets the adiabatic index for the gas to have the value of \p adi_index.
  /// \param[in] adi_index The adiabatic index of the gas.
  ripple_host_device constexpr IdealGas(value_t adi_index)
  : _adi_index(adi_index) {}

  /// Returns a reference to the adiabatic index for the ideal gas.
  ripple_host_device constexpr auto adi() -> value_t& {
    return _adi_index;
  }

  /// Returns a const reference to the adiabatic index for the ideal gas.
  ripple_host_device constexpr auto adi() const -> const value_t& {
    return _adi_index;
  }

  /// Evaluates the equation of state for the ideal gas, which is given by:
  ///
  /// \begin{equation}
  ///   e = e(p, \rho) = \frac{p}{(\gamma - 1) \rho}
  /// \end{equation}
  /// 
  /// and returns the result.
  /// \param  state The state to use to evaluate the quation of state.
  /// \tparam State The type of the state.
  template <typename State>
  ripple_host_device constexpr auto eos(const State& state) const 
  -> value_t {
    return state.pressure(*this) / ((_adi_index - _1) * state.rho());
  }

  /// Calculates the speed of sound for the ideal gas, based on the equation
  /// of state, where the sound speed is given by:
  ///
  /// \begin{equation}
  ///   a = \sqrt{\frac{\gamma p}{\rho}}
  /// \end{equation}
  ///
  /// and returns the result.
  /// \param  state   The state to use to compute the sound speed.
  /// \tparam State   The type of the state.
  template <typename State>
  ripple_host_device constexpr auto sound_speed(const State& state) const 
  -> value_t {
    return std::sqrt(_adi_index * state.pressure(*this) / state.rho());
  } 

 private:
  value_t _adi_index = 1.4; //!< The adiabatic index for the gas.
};

} // namespace ripple::fv

#endif // RIPPLE_EOS_IDEAL_GAS_HPP


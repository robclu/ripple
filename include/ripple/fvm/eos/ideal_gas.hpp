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

  /// Defines the type of the equation of state.
  using self_t = IdealGas;

 public:
  /// Defines the type of the data used by the material.
  using value_t = std::decay_t<T>;

  //==--- [construction] ---------------------------------------------------==//

  /// Sets the value of the gas to have the default adiabatic index of 1.4.
  constexpr IdealGas() = default;

  /// Sets the adiabatic index for the gas to have the value of \p adi_index.
  /// \param[in] adi_index The adiabatic index of the gas.
  ripple_host_device constexpr IdealGas(value_t adi_index)
  : _adi_index(adi_index) {}

  /// Constructs the equation of state from the \p other equation of state.
  /// \param other The other equation of state.
  ripple_host_device constexpr IdealGas(const IdealGas& other)
  : _adi_index{other._adi_index} {}

  /// Constructs the equation of state from the \p other equation of state,
  /// moving the \p other ideal gas into this one.
  /// \param other The other equation of state.
  ripple_host_device constexpr IdealGas(IdealGas&& other)
  : _adi_index{std::move(other._adi_index)} {}
  
  //==--- [operator overloads] ---------------------------------------------==//
  
  /// Overload of copy assignment overload to create the equation of state from
  /// the \p other equation of state.
  /// \param other The other equation of state to copy.
  ripple_host_device constexpr auto operator=(const IdealGas& other) 
  -> self_t& {
    _adi_index = other.adi();
    return *this;
  }

  /// Overload of move assignment overload to create the equation of state from
  /// the \p other equation of state.
  /// \param other The other equation of state to move.
  ripple_host_device constexpr auto operator=(IdealGas&& other) 
  -> self_t& {
    _adi_index = std::move(other._adi_index);
    return *this;
  }

  /// Overload of copy assignment overload to create the equation of state from
  /// the \p other equation of state.
  /// \param other The other equation of state to copy.
  ripple_host_device constexpr auto operator=(const Eos<IdealGas>& other) 
  -> self_t& {
    _adi_index = other.adi();
    return *this;
  }

  /// Overload of comparison opertor to compare this equation of state against
  /// the \p other equation of state. They are equal if they have the same type
  /// and same adiabatic index.
  /// \param  other The other equation of state to compare against.
  /// \tparam Other The type of the other equation of state.
  template <typename OtherImpl>
  ripple_host_device constexpr auto operator==(const Eos<OtherImpl>& other)
  const -> bool {
    return std::is_same_v<std::decay_t<OtherImpl>, self_t> &&
      other.adi() == _adi_index;
  }
 
  //==--- [interface] ------------------------------------------------------==// 

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

  /// Returns the name of the equation of sate.
  ripple_host_device constexpr auto name() const -> const char* {
    return eos_traits_t<self_t>::name;
  }

 private:
  value_t _adi_index = 1.4; //!< The adiabatic index for the gas.
};

} // namespace ripple::fv

#endif // RIPPLE_EOS_IDEAL_GAS_HPP


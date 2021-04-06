/**=--- ripple/benchmarks/force.hpp ------------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  eos.hpp
 * \brief This file implements and equations of state for an ideal gas.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARK_FLUX_EOS_HPP
#define RIPPLE_BENCHMARK_FLUX_EOS_HPP

#include <ripple/utility/portability.hpp>
#include <cmath>
#include <type_traits>

/**
 * The IdealGas type implements the equation of state interface which
 * represents an ideal gas.
 * \tparam T  The type of the data for the ideal gas.
 */
template <typename T>
struct IdealGas {
 public:
  /** Defines the type of the data used for the gas.. */
  using ValueType = std::remove_reference_t<T>;

  /**
   * Default constructor.
   */
  ripple_host_device constexpr IdealGas() = default;

  /**
   * Constructor to set the adiabatic index.
   * \param adi_index The adiabatic index.
   */
  ripple_host_device constexpr IdealGas(ValueType adi_index) noexcept
  : adi_index_{adi_index} {}

  /**
   * Returns a reference to the adiabatic index for the ideal gas.
   */
  ripple_host_device constexpr auto adi() noexcept -> ValueType& {
    return adi_index_;
  }

  /**
   * Returns a const reference to the adiabatic index for the ideal gas.
   */
  ripple_host_device constexpr auto adi() const noexcept -> ValueType {
    return adi_index_;
  }

  /**
   * Evaluates the equation of state for the ideal gas, which is given by:
   *
   * \begin{equation}
   *   e = e(p, \rho) = \frac{p}{(\gamma - 1) \rho}
   * \end{equation}
   *
   * and returns the result.
   * \param  state The state to use to evaluate the quation of state.
   * \tparam State The type of the state.
   */
  template <typename State>
  ripple_host_device constexpr auto
  eos(const State& state) const noexcept -> ValueType {
    return state.pressure(*this) / ((adi_index_ - ValueType(1)) * state.rho());
  }

  /**
   * Calculates the speed of sound for the ideal gas, based on the equation
   * of state, where the sound speed is given by:
   *
   * \begin{equation}
   *   a = \sqrt{\frac{\gamma p}{\rho}}
   * \end{equation}
   *
   * and returns the result.
   * \param  state   The state to use to compute the sound speed.
   * \tparam State   The type of the state.
   */
  template <typename State>
  ripple_host_device constexpr auto
  sound_speed(const State& state) const noexcept -> ValueType {
    return std::sqrt(adi_index_ * state.pressure(*this) / state.rho());
  }

 private:
  ValueType adi_index_ = 1.4; //!< The adiabatic index for the gas.
};

#endif
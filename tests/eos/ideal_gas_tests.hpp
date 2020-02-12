//==--- ripple/tests/eos/ideal_gas_tests.hpp --------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  ideal_gas_tests.hpp
/// \brief This file defines tests for an ideal gas equation of state.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_EOS_IDEAL_GAS_TESTS_HPP
#define RIPPLE_TESTS_EOS_IDEAL_GAS_TESTS_HPP

#include <ripple/fvm/eos/ideal_gas.hpp>
#include <ripple/fvm/state/fluid_state.hpp>
#include <gtest/gtest.h>

using real_t       = double;
using ideal_gas_t  = ripple::fv::IdealGas<real_t>;
using test_state_t = ripple::fv::FluidState<real_t, ripple::Num<1>>;

TEST(eos_ideal_gas, computes_eos_correctly) {
  const auto rho = real_t{1};
  const auto p   = real_t{1};
  const auto adi = real_t{1.4};

  test_state_t s;
  ideal_gas_t  gas(adi);

  s.rho() = rho;
  s.set_v(ripple::dim_x, real_t{1});
  s.set_pressure(p, gas);

  // Eos is : p / ((adi - 1) * rho)
  const auto eos_val = p / (adi - 1.0) * rho;
  EXPECT_EQ(gas.eos(s), eos_val);
}

TEST(eos_ideal_gas, computes_speed_of_sound_correctly) {
  const auto rho = real_t{1};
  const auto p   = real_t{1};
  const auto adi = real_t{1.4};

  test_state_t s;
  ideal_gas_t  gas(adi);

  s.rho() = rho;
  s.set_v(ripple::dim_x, real_t{1});
  s.set_pressure(p, gas);

  // Eound speed is : sqrt(adi * p / rho)
  const auto sound_speed = std::sqrt(adi * p / rho);
  EXPECT_EQ(gas.sound_speed(s), sound_speed);
}

#endif // RIPPLE_TESTS_EOS_IDEAL_GAS_TESTS_HPP

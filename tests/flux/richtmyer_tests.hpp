//==--- ripple/tests/flux/richtmyer_tests.hpp -------------- -*- C++ -*- ---==//
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

#ifndef RIPPLE_TESTS_FLUX_RICHTMYER_TESTS_HPP
#define RIPPLE_TESTS_FLUX_RICHTMYER_TESTS_HPP

#include <ripple/fvm/eos/ideal_gas.hpp>
#include <ripple/fvm/flux/richtmyer.hpp>
#include <ripple/fvm/state/fluid_state.hpp>
#include <ripple/fvm/state/state_initializer.hpp>
#include <gtest/gtest.h>

using namespace ripple;
using namespace ripple::fv;
using namespace ripple::fv::state;

using real_t           = double;
using ideal_gas_t      = IdealGas<real_t>;
using fluid_state_1d_t = FluidState<real_t, Num<1>>;
using richmyer_t       = Richtmyer;

TEST(flux_richtmyer, computes_flux_correctly_1d) {
  constexpr auto dt = real_t{0.03}, dh = real_t{0.1};
  fluid_state_1d_t l; fluid_state_1d_t r;
  auto g = ideal_gas_t(1.4);
  auto flux = richmyer_t();
 
  auto l_init = make_state_initializer(1.000_rho, 1.0_p, 0.75_v_x);
  auto r_init = make_state_initializer(0.125_rho, 0.1_p, 0.00_v_x);

  l_init.set_state_data(l, g); r_init.set_state_data(r, g);

  EXPECT_EQ(l.rho()   , real_t{1.0});
  EXPECT_EQ(l.v(dim_x), real_t{0.75});

  auto res = flux.evaluate(l, r, g, dim_x, dt, dh);

  EXPECT_NEAR(res[0], 0.59437500, 1e-7);
  EXPECT_NEAR(res[1], 2.30066786, 1e-7);
  EXPECT_NEAR(res[2], 1.19511042, 1e-7);
}

#endif // RIPPLE_TESTS_FLUX_RICHTMYER_TESTS_HPP


//==--- ripple/tests/state/state_initializer_tests.hpp ----- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_initializer_tests.hpp
/// \brief This file defines tests for the state initializer.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STATE_STATE_INITIALIZER_TESTS_HPP
#define RIPPLE_TESTS_STATE_STATE_INITIALIZER_TESTS_HPP

#include <ripple/fvm/state/state_initializer.hpp>
#include <gtest/gtest.h>

using namespace ripple::fv::state;

using real_it      = double;
using fs_1d_it     = ripple::fv::FluidState<real_t, ripple::Num<1>>;
using fs_2d_it     = ripple::fv::FluidState<real_t, ripple::Num<2>>;
using fs_3d_it     = ripple::fv::FluidState<real_t, ripple::Num<3>>;
using ideal_gas_it = ripple::fv::IdealGas<real_t>;

TEST(state_state_initialzier, can_initializer_fluid_state_1d) {
  auto initializer = ripple::fv::make_state_initializer(
    1.5_rho, -2.2_v_x, 3.0_p
  );

  fs_1d_t s;
  ideal_gas_it g(1.4);

  initializer.set_state_data(s, g);

  EXPECT_EQ(s.rho()           , real_it{1.5} );
  EXPECT_EQ(s.pressure(g)     , real_it{3.0} );
  EXPECT_EQ(s.v(ripple::dim_x), real_it{-2.2});
}

TEST(state_state_initialzier, can_initializer_fluid_state_2d) {
  constexpr real_it tol = 1e-6;
  auto initializer = ripple::fv::make_state_initializer(
    1.5_rho, -2.2_v_x, 3.0_p, 3.2_v_y
  );

  fs_2d_t s;
  ideal_gas_it g(1.4);

  initializer.set_state_data(s, g);

  EXPECT_NEAR(s.rho()           , real_it{1.5} , tol);
  EXPECT_NEAR(s.pressure(g)     , real_it{3.0} , tol);
  EXPECT_NEAR(s.v(ripple::dim_x), real_it{-2.2}, tol);
  EXPECT_NEAR(s.v(ripple::dim_y), real_it{3.2} , tol);
}

TEST(state_state_initialzier, can_initializer_fluid_state_3d) {
  constexpr real_it tol = 1e-6;
  auto initializer = ripple::fv::make_state_initializer(
    1.5_rho, -2.2_v_x, 3.0_p, 3.2_v_y, -1.3_v_z
  );

  fs_3d_t s;
  ideal_gas_it g(1.4);

  initializer.set_state_data(s, g);

  EXPECT_NEAR(s.rho()           , real_it{1.5} , tol);
  EXPECT_NEAR(s.pressure(g)     , real_it{3.0} , tol);
  EXPECT_NEAR(s.v(ripple::dim_x), real_it{-2.2}, tol);
  EXPECT_NEAR(s.v(ripple::dim_y), real_it{3.2} , tol);
  EXPECT_NEAR(s.v(ripple::dim_z), real_it{-1.3}, tol);
}

#endif // RIPPLE_TESTS_STATE_STATE_INITIALIZER_TESTS_HPP


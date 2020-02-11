//==--- ripple/tests/state/fluid_state_tests.hpp ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fluid_state_tests.hpp
/// \brief This file defines tests for fluid state functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STATE_FLUID_STATE_TESTS_HPP
#define RIPPLE_TESTS_STATE_FLUID_STATE_TESTS_HPP

#include <ripple/fvm/state/fluid_state.hpp>
#include <gtest/gtest.h>

using real_t  = float;
using fs_1d_t = ripple::fv::FluidState<real_t, ripple::Num<1>>;
using fs_2d_t = ripple::fv::FluidState<real_t, ripple::Num<2>>;
using fs_3d_t = ripple::fv::FluidState<real_t, ripple::Num<3>>;

TEST(state_fluid_state, can_create_default_state) {
  fs_1d_t s1;
  fs_2d_t s2;
  fs_3d_t s3;

  EXPECT_EQ(s1.size(), size_t{3});
  EXPECT_EQ(s2.size(), size_t{4});
  EXPECT_EQ(s3.size(), size_t{5});
}

TEST(state_fluid_state, can_set_and_modify_density) {
  fs_3d_t s3;

  s3.rho() = 2.2f;
  EXPECT_EQ(s3.rho(), 2.2f);
  s3.rho() = 4.4f;
  EXPECT_EQ(s3.rho(), 4.4f);
}

TEST(state_fluid_state, can_set_and_modify_energy) {
  fs_3d_t s;

  s.energy() = 2.2f;
  EXPECT_EQ(s.energy(), 2.2f);
  s.energy() = 4.4f;
  EXPECT_EQ(s.energy(), 4.4f);
}

TEST(state_fluid_state, can_set_and_modify_density_velocity) {
  fs_3d_t s;

  s.rho() = 2.2f;
  EXPECT_EQ(s.rho(), 2.2f);
  s.rho() = 4.4f;
  EXPECT_EQ(s.rho(), 4.4f);
}

TEST(state_fluid_state, can_set_and_modify_velocity) {
  fs_3d_t s;
  s.rho() = 2.0f;

  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    constexpr auto dim = size_t{d};
    s.set_v<dim>(3.0f);

    EXPECT_EQ(s.v<dim>()    , 3.0f);
    EXPECT_EQ(s.v(dim)      , 3.0f);
    EXPECT_EQ(s.rho_v<dim>(), 3.0f * 2.0f);
    EXPECT_EQ(s.rho_v(dim)  , 3.0f * 2.0f);
  });
}

//==--- [array operations on state] ----------------------------------------==//

TEST(state_fluid_state, can_add_states) {
  fs_3d_t s;
  s.rho()    = 2.0f;
  s.energy() = 3.0f;
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, 4.0f);
  });

  s += s;

  EXPECT_EQ(s.rho()   , 4.0f);
  EXPECT_EQ(s.energy(), 6.0f);

  // State stores rho_v, so since rho * 2, v stays the same.
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    EXPECT_EQ(s.v(d)    , 4.0f);
    EXPECT_EQ(s.rho_v(d), 16.0f);
  });
}

TEST(state_fluid_state, can_subtract_states) {
  fs_3d_t s;
  s.rho()    = 2.0f;
  s.energy() = 3.0f;
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, 4.0f);
  });

  auto s1 = s + s;
  s1 -= s;

  EXPECT_EQ(s1.rho()   , 2.0f);
  EXPECT_EQ(s1.energy(), 3.0f);

  ripple::unrolled_for<s1.dimensions()>([&] (auto d) {
    EXPECT_EQ(s1.v(d)    , 4.0f);
    EXPECT_EQ(s1.rho_v(d), 8.0f);
  });
}

TEST(state_fluid_state, can_multiply_states) {
  fs_3d_t s;
  s.rho()    = 2.0f;
  s.energy() = 3.0f;
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, 4.0f);
  });

  s *= s;

  EXPECT_EQ(s.rho()   , 4.0f);
  EXPECT_EQ(s.energy(), 9.0f);

  // this is rho_v * rho_v / rho:
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    EXPECT_EQ(s.v(d)    , 16.0f);
    EXPECT_EQ(s.rho_v(d), 64.0f);
  });
}

TEST(state_fluid_state, can_divide_states) {
  fs_3d_t s;
  s.rho()    = 2.0f;
  s.energy() = 3.0f;
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, 4.0f);
  });

  auto s1 = s * s;
  s1 /= s;

  EXPECT_EQ(s1.rho()   , 2.0f);
  EXPECT_EQ(s1.energy(), 3.0f);

  ripple::unrolled_for<s1.dimensions()>([&] (auto d) {
    EXPECT_EQ(s1.v(d)    , 4.0f);
    EXPECT_EQ(s1.rho_v(d), 8.0f);
  });
}

#endif // RIPPLE_TESTS_STATE_FLUID_STATE_TESTS_HPP

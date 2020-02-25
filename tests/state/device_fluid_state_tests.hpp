//==--- ripple/tests/state/device_fluid_state_tests.hpp ---- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_fluid_state_tests_device.hpp
/// \brief This file defines tests for fluid state functionality on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_STATE_DEVICE_FLUID_STATE_TESTS_HPP
#define RIPPLE_TESTS_STATE_DEVICE_FLUID_STATE_TESTS_HPP

#include <ripple/core/container/grid.hpp>
#include <ripple/fvm/eos/ideal_gas.hpp>
#include <ripple/fvm/state/fluid_state.hpp>
#include <ripple/fvm/state/state_initializer.hpp>
#include <gtest/gtest.h>

using real_dt      = double;
using fs_1d_dt     = ripple::fv::FluidState<real_dt, ripple::Num<1>>;
using fs_2d_dt     = ripple::fv::FluidState<real_dt, ripple::Num<2>>;
using fs_3d_dt     = ripple::fv::FluidState<real_dt, ripple::Num<3>>;
using ideal_gas_dt = ripple::fv::IdealGas<real_dt>;

TEST(state_fluid_state, can_init_states_in_grid) {
  constexpr size_t dims   = 3;
  constexpr size_t size_x = 30;
  constexpr size_t size_y = 30;
  constexpr size_t size_z = 30;

  using states_t = ripple::grid_3d_t<fs_3d_dt>;

  auto topo = ripple::Topology();
  states_t states(topo, size_x, size_y, size_z);

  auto initializer = ripple::make_pipeline(
    [] ripple_host_device (auto state) {
      state->rho()    = real_dt{2};
      state->energy() = real_dt{3};

      ripple::unrolled_for<dims>([&] (auto d) {
        state->set_v(d, real_dt{4});
      });
    }
  );

  states.apply_pipeline(initializer);

  for (auto k : ripple::range(states.size(ripple::dim_z))) {
    for (auto j : ripple::range(states.size(ripple::dim_y))) {
      for (auto i : ripple::range(states.size(ripple::dim_x))) {
        auto state = states(i, j, k);
        EXPECT_EQ(state->rho()   , real_dt{2});
        EXPECT_EQ(state->energy(), real_dt{3});
        for (auto dim : ripple::range(state->dimensions())) {
          EXPECT_EQ(state->v(dim)    , real_dt{4});
          EXPECT_EQ(state->rho_v(dim), real_dt{8});
        }
      }
    }  
  }
}

TEST(state_fluid_state, can_init_states_in_grid_with_initializer) {
  using states_t = ripple::grid_3d_t<fs_3d_dt>;
  using namespace ripple::fv::state;

  constexpr size_t dims   = 3;
  constexpr size_t size_x = 30;
  constexpr size_t size_y = 30;
  constexpr size_t size_z = 30;
  constexpr real_dt tol   = 1e-6;

  auto initializer = ripple::fv::make_state_initializer(
    1.5_rho, -2.2_v_x, 3.0_p, 3.2_v_y, -1.3_v_z
  );

  auto topo = ripple::Topology();
  states_t     states(topo, size_x, size_y, size_z);
  ideal_gas_dt gas(1.4);

  auto setter = ripple::make_pipeline(
    ripple::make_invocable(
      [] ripple_host_device (auto state, auto& init, auto& eos) {
        init.set_state_data(*state, eos);
      }, initializer, gas
    )
  );

  states.apply_pipeline(setter);

  for (auto k : ripple::range(states.size(ripple::dim_z))) {
    for (auto j : ripple::range(states.size(ripple::dim_y))) {
      for (auto i : ripple::range(states.size(ripple::dim_x))) {
        auto s = states(i, j, k);

        EXPECT_NEAR(s->rho()           , real_dt{1.5} , tol);
        EXPECT_NEAR(s->pressure(gas)   , real_dt{3.0} , tol);
        EXPECT_NEAR(s->v(ripple::dim_x), real_dt{-2.2}, tol);
        EXPECT_NEAR(s->v(ripple::dim_y), real_dt{3.2} , tol);
        EXPECT_NEAR(s->v(ripple::dim_z), real_dt{-1.3}, tol);
      }
    }  
  }

 
}

#endif // RIPPLE_TESTS_STATE_DEVICE_FLUID_STATE_TESTS_H

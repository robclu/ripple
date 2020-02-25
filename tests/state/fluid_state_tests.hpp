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

#include <ripple/core/container/grid.hpp>
#include <ripple/core/execution/dynamic_execution_params.hpp>
#include <ripple/core/functional/invoke.hpp>
#include <ripple/fvm/eos/ideal_gas.hpp>
#include <ripple/fvm/state/fluid_state.hpp>
#include <gtest/gtest.h>

using real_t       = double;
using fs_1d_t      = ripple::fv::FluidState<real_t, ripple::Num<1>>;
using fs_2d_t      = ripple::fv::FluidState<real_t, ripple::Num<2>>;
using fs_3d_t      = ripple::fv::FluidState<real_t, ripple::Num<3>>;
using ideal_gas_st = ripple::fv::IdealGas<real_t>;

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

  s3.rho() = real_t{2.2};
  EXPECT_EQ(s3.rho(), real_t{2.2});
  s3.rho() = real_t{4.4};
  EXPECT_EQ(s3.rho(), real_t{4.4});
}

TEST(state_fluid_state, can_set_and_modify_energy) {
  fs_3d_t s;

  s.energy() = real_t{2.2};
  EXPECT_EQ(s.energy(), real_t{2.2});
  s.energy() = real_t{4.4};
  EXPECT_EQ(s.energy(), real_t{4.4});
}

TEST(state_fluid_state, can_set_and_modify_density_velocity) {
  fs_3d_t s;

  s.rho() = real_t{2.2};
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    constexpr auto dim = size_t{d};
    constexpr auto v   = real_t{3};
    s.rho_v<dim>() = v * s.rho();

    EXPECT_EQ(s.v<dim>()    , v);
    EXPECT_EQ(s.v(dim)      , v);
    EXPECT_EQ(s.rho_v<dim>(), v * s.rho());
    EXPECT_EQ(s.rho_v(dim)  , v * s.rho());
  });
  EXPECT_EQ(s.rho(), real_t{2.2});
}

TEST(state_fluid_state, can_set_and_modify_velocity) {
  fs_3d_t s;
  s.rho() = real_t{2};

  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    constexpr auto dim = size_t{d};
    constexpr auto v   = real_t{3};
    s.set_v<dim>(v);

    EXPECT_EQ(s.v<dim>()    , v);
    EXPECT_EQ(s.v(dim)      , v);
    EXPECT_EQ(s.rho_v<dim>(), v * real_t{2});
    EXPECT_EQ(s.rho_v(dim)  , v * real_t{2});
  });
}

TEST(state_fluid_state, can_get_and_set_pressure) {
  fs_3d_t s;
  ideal_gas_st gas(real_t{1.4});

  s.rho()    = real_t{2};
  s.energy() = real_t{1};
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    constexpr auto dim = size_t{d};
    s.set_v<dim>(real_t{3});
  });

  s.set_pressure(real_t{2.5}, gas);

  EXPECT_LT(std::abs(s.pressure(gas) - real_t{2.5}), real_t{1e-6});
}

TEST(state_fluid_state, computes_fluxes_correctly_1d) {
  constexpr auto rho    = real_t{1};
  constexpr auto u      = real_t{2};
  constexpr auto energy = real_t{3};
  constexpr auto adi    = real_t{1.4};

  fs_1d_t s;
  ideal_gas_st gas(adi);

  // Explicitly compute the pressure:
  constexpr auto p = (adi - real_t{1}) * (energy - real_t{.5} * rho * u * u);

  s.rho()    = rho;
  s.energy() = energy;
  s.set_v(ripple::dim_x, u);

  EXPECT_EQ(s.pressure(gas), p);

  // Explicit computations for the flux:
  const auto flux = s.flux(gas, ripple::dim_x);
  EXPECT_EQ(flux[0], rho * u         ); // Density component
  EXPECT_EQ(flux[1], u * (energy + p)); // Energy component
  EXPECT_EQ(flux[2], rho * u * u + p ); // v_x component
}

TEST(state_fluid_state, computes_fluxes_correctly_2d) {
  constexpr auto rho    = real_t{1};
  constexpr auto u      = real_t{2};
  constexpr auto v      = real_t{2.5};
  constexpr auto energy = real_t{3};
  constexpr auto adi    = real_t{1.4};

  fs_2d_t s;
  ideal_gas_st gas(adi);

  constexpr auto uu = u * u;
  constexpr auto vv = v * v;

  // Explicitly compute the pressure:
  constexpr auto p = 
    (adi - real_t{1}) * (energy - real_t{.5} * rho * (uu + vv));

  s.rho()    = rho;
  s.energy() = energy;
  s.set_v(ripple::dim_x, u);
  s.set_v(ripple::dim_y, v);

  EXPECT_EQ(s.v(ripple::dim_x), u);
  EXPECT_EQ(s.v(ripple::dim_y), v);
  EXPECT_EQ(s.v<0>()          , u);
  EXPECT_EQ(s.v<1>()          , v);
  EXPECT_EQ(s.pressure(gas)   , p);

  // Explicit computations for the flux:
  const auto flux_x = s.flux(gas, ripple::dim_x);
  EXPECT_EQ(flux_x[0], rho * u         ); // Density component 
  EXPECT_EQ(flux_x[1], u * (energy + p)); // Energy component
  EXPECT_EQ(flux_x[2], rho * u * u + p ); // v_x component
  EXPECT_EQ(flux_x[3], rho * u * v     ); // v_y component

  const auto flux_y = s.flux(gas, ripple::dim_y);
  EXPECT_EQ(flux_y[0], rho * v         ); // Density component 
  EXPECT_EQ(flux_y[1], v * (energy + p)); // Energy component
  EXPECT_EQ(flux_y[2], rho * u * v     ); // v_x component
  EXPECT_EQ(flux_y[3], rho * v * v + p ); // v_y component
}

TEST(state_fluid_state, computes_fluxes_correctly_3d) {
  constexpr auto rho    = real_t{1};
  constexpr auto u      = real_t{2};
  constexpr auto v      = real_t{2.5};
  constexpr auto w      = real_t{3.5};
  constexpr auto energy = real_t{3};
  constexpr auto adi    = real_t{1.4};

  fs_3d_t s;
  ideal_gas_st gas(adi);

  constexpr auto uu = u * u;
  constexpr auto vv = v * v;
  constexpr auto ww = w * w;

  // Explicitly compute the pressure:
  constexpr auto p = 
    (adi - real_t{1}) * (energy - real_t{.5} * rho * (uu + vv + ww));

  s.rho()    = rho;
  s.energy() = energy;
  s.set_v(ripple::dim_x, u);
  s.set_v(ripple::dim_y, v);
  s.set_v(ripple::dim_z, w);

  EXPECT_EQ(s.v(ripple::dim_x), u);
  EXPECT_EQ(s.v(ripple::dim_y), v);
  EXPECT_EQ(s.v(ripple::dim_z), w);
  EXPECT_EQ(s.v<0>()          , u);
  EXPECT_EQ(s.v<1>()          , v);
  EXPECT_EQ(s.v<2>()          , w);
  EXPECT_EQ(s.pressure(gas)   , p);

  // Explicit computations for the flux:
  const auto flux_x = s.flux(gas, ripple::dim_x);
  EXPECT_EQ(flux_x[0], rho * u         ); // Density component 
  EXPECT_EQ(flux_x[1], u * (energy + p)); // Energy component
  EXPECT_EQ(flux_x[2], rho * u * u + p ); // v_x component
  EXPECT_EQ(flux_x[3], rho * u * v     ); // v_y component
  EXPECT_EQ(flux_x[4], rho * u * w     ); // v_z component

  const auto flux_y = s.flux(gas, ripple::dim_y);
  EXPECT_EQ(flux_y[0], rho * v         ); // Density component 
  EXPECT_EQ(flux_y[1], v * (energy + p)); // Energy component
  EXPECT_EQ(flux_y[2], rho * v * u     ); // v_x component
  EXPECT_EQ(flux_y[3], rho * v * v + p ); // v_y component
  EXPECT_EQ(flux_y[4], rho * v * w     ); // v_y component

  const auto flux_z = s.flux(gas, ripple::dim_z);
  EXPECT_EQ(flux_z[0], rho * w         ); // Density component 
  EXPECT_EQ(flux_z[1], w * (energy + p)); // Energy component
  EXPECT_EQ(flux_z[2], rho * w * u     ); // v_x component
  EXPECT_EQ(flux_z[3], rho * w * v     ); // v_y component
  EXPECT_EQ(flux_z[4], rho * w * w + p ); // v_z component
}

//==--- [array operations on state] ----------------------------------------==//

TEST(state_fluid_state, can_add_states) {
  fs_3d_t s;
  constexpr auto v = real_t{4};

  s.rho()    = real_t{2};
  s.energy() = real_t{3};
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, v);
  });

  s += s;

  EXPECT_EQ(s.rho()   , real_t{4});
  EXPECT_EQ(s.energy(), real_t{6});

  // State stores rho_v, so since rho * 2, v stays the same.
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    EXPECT_EQ(s.v(d)    , v);
    EXPECT_EQ(s.rho_v(d), real_t{16});
  });
}

TEST(state_fluid_state, can_subtract_states) {
  fs_3d_t s;
  constexpr auto v = real_t{4};

  s.rho()    = real_t{2};
  s.energy() = real_t{3};
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, v);
  });

  auto s1 = s + s;
  s1 -= s;

  EXPECT_EQ(s1.rho()   , real_t{2});
  EXPECT_EQ(s1.energy(), real_t{3});

  ripple::unrolled_for<s1.dimensions()>([&] (auto d) {
    EXPECT_EQ(s1.v(d)    , v);
    EXPECT_EQ(s1.rho_v(d), v * 2);
  });
}

TEST(state_fluid_state, can_multiply_states) {
  fs_3d_t s;
  constexpr auto v = real_t{4};

  s.rho()    = real_t{2};
  s.energy() = real_t{3};
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, v);
  });

  s *= s;

  EXPECT_EQ(s.rho()   , real_t{4});
  EXPECT_EQ(s.energy(), real_t{9});

  // this is rho_v * rho_v / rho:
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    EXPECT_EQ(s.v(d)    , real_t{16});
    EXPECT_EQ(s.rho_v(d), real_t{64});
  });
}

TEST(state_fluid_state, can_divide_states) {
  fs_3d_t s;
  s.rho()    = real_t{2};
  s.energy() = real_t{3};
  ripple::unrolled_for<s.dimensions()>([&] (auto d) {
    s.set_v(d, real_t{4});
  });

  auto s1 = s * s;
  s1 /= s;

  EXPECT_EQ(s1.rho()   , real_t{2});
  EXPECT_EQ(s1.energy(), real_t{3});

  ripple::unrolled_for<s1.dimensions()>([&] (auto d) {
    EXPECT_EQ(s1.v(d)    , real_t{4});
    EXPECT_EQ(s1.rho_v(d), real_t{8});
  });
}

//==-- [state inside grid] -------------------------------------------------==//

#endif // RIPPLE_TESTS_STATE_FLUID_STATE_TESTS_HPP

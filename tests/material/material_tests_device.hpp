//==--- ripple/core/tests/material/material_tests_device.hpp -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_tests_device.hpp
/// \brief This file contains all material device tests.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_MATERIAL_MATERIAL_TESTS_DEVICE_HPP
#define RIPPLE_TESTS_MATERIAL_MATERIAL_TESTS_DEVICE_HPP

#include <ripple/fvm/eos/ideal_gas.hpp>
#include <ripple/fvm/levelset/levelset.hpp>
#include <ripple/fvm/material/material.hpp>
#include <ripple/fvm/state/fluid_state.hpp>
#include <ripple/fvm/state/state_initializer.hpp>
#include <gtest/gtest.h>

using real_t      = double;
using dim3_t      = ripple::Num<3>;
using state_t     = ripple::fv::FluidState<real_t, dim3_t>;
using ideal_gas_t = ripple::fv::IdealGas<real_t>;
using levelset_t  = ripple::fv::Levelset<real_t, dim3_t>;
using material_t  = ripple::fv::Material<state_t, levelset_t, ideal_gas_t>;

using namespace ripple::fv::state;

constexpr real_t tol = 1e-8;

TEST(material_tests, can_create_and_initialize_material) {
  namespace fvm = ripple::fv;
  constexpr size_t size_x = 52;
  constexpr size_t size_y = 71;
  constexpr size_t size_z = 49;
  constexpr real_t r      = 0.25;

  auto topo = ripple::Topology();
  auto gas  = ideal_gas_t(1.4);

  material_t material(topo, gas, size_x, size_y, size_z);

  material.initialize(
    [] ripple_host_device (auto levelset) {
      const auto nx = ripple::global_norm_idx(ripple::dim_x);
      const auto ny = ripple::global_norm_idx(ripple::dim_y);
      const auto nz = ripple::global_norm_idx(ripple::dim_z);

      *levelset = std::sqrt(nx * nx + ny * ny + nz * nz) - r;
    },
    fvm::make_state_initializer(0.1_rho, 0.2_p, 0.3_v_x, 0.4_v_y, 0.5_v_z),
    fvm::make_state_initializer(1.1_rho, 1.2_p, 1.3_v_x, 1.4_v_y, 1.5_v_z)
  );

  for (auto k : ripple::range(material.size(ripple::dim_z))) {
    for (auto j : ripple::range(material.size(ripple::dim_y))) {
      for (auto i : ripple::range(material.size(ripple::dim_x))) {
        const auto nx = static_cast<real_t>(i) / size_x;
        const auto ny = static_cast<real_t>(j) / size_y;
        const auto nz = static_cast<real_t>(k) / size_z;

        const auto inside = (std::sqrt(nx * nx + ny * ny + nz * nz) - r) <= 0.0;

        auto m = material(i, j, k);
        EXPECT_EQ(m.levelset()->inside()     , inside);
        EXPECT_NEAR(m.state()->rho()           , inside ? 0.1 : 1.1, tol);
        EXPECT_NEAR(m.state()->pressure(gas)   , inside ? 0.2 : 1.2, tol);
        EXPECT_NEAR(m.state()->v(ripple::dim_x), inside ? 0.3 : 1.3, tol);
        EXPECT_NEAR(m.state()->v(ripple::dim_y), inside ? 0.4 : 1.4, tol);
        EXPECT_NEAR(m.state()->v(ripple::dim_z), inside ? 0.5 : 1.5, tol);
      }
    }
  }
}

#endif // RIPPLE_TESTS_MATERIAL_MATERIAL_TESTS_DEVICE_HPP

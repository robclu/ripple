//==--- ripple/core/tests/container/device_grid_tests.hpp -- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_grid_tests.hpp
/// \brief This file defines tests for device side grid functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_DEVICE_GRID_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_DEVICE_GRID_TESTS_HPP

#include <ripple/core/container/grid.hpp>

// Defines the data type for the tests.
using real_t = float;

TEST(container_grid, can_apply_pipeline_on_device) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, size_x);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x));
    }
  );
  g.apply_pipeline(pipeline);

  for (auto i : ripple::range(g.size())) {
    EXPECT_EQ(*g(i), static_cast<real_t>(i));
  }
}

#endif // RIPPLE_TESTS_CONTAINER_DEVICE_GRID_TESTS_HPP


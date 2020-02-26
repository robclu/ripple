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

TEST(container_grid, can_apply_pipeline_on_device_1d) {
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

TEST(container_grid, can_apply_pipeline_on_device_2d) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g(topo, size_x, size_y);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x))
          + static_cast<real_t>(global_idx(ripple::dim_y));
    }
  );
  g.apply_pipeline(pipeline);

  for (auto j : ripple::range(g.size(ripple::dim_y))) {
    for (auto i : ripple::range(g.size(ripple::dim_x))) {
      EXPECT_EQ(*g(i, j), static_cast<real_t>(i + j));
    }
  }
}

TEST(container_grid, can_apply_pipeline_on_device_3d) {
  constexpr auto size_x = size_t{100};
  constexpr auto size_y = size_t{100};
  constexpr auto size_z = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g(topo, size_x, size_y, size_z);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = static_cast<real_t>(global_idx(ripple::dim_x))
          + static_cast<real_t>(global_idx(ripple::dim_y))
          + static_cast<real_t>(global_idx(ripple::dim_z));
    }
  );
  g.apply_pipeline(pipeline);

  for (auto k : ripple::range(g.size(ripple::dim_z))) {
    for (auto j : ripple::range(g.size(ripple::dim_y))) {
      for (auto i : ripple::range(g.size(ripple::dim_x))) {
        EXPECT_EQ(*g(i, j, k), static_cast<real_t>(i + j + k));
      }
    }
  }
}

//==--- [normalizes indices] -----------------------------------------------==//

TEST(container_grid, can_get_normalized_global_index_1d) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, size_x);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = global_norm_idx(ripple::dim_x);
    }
  );
  g.apply_pipeline(pipeline);

  for (auto i : ripple::range(g.size())) {
    const auto norm_idx = static_cast<real_t>(i) / size_x;
    EXPECT_EQ(*g(i), norm_idx);
  }
}

TEST(container_grid, can_get_normalized_global_index_2d) {
  constexpr auto size_x = size_t{101};
  constexpr auto size_y = size_t{101};
  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g(topo, size_x, size_y);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = global_norm_idx(ripple::dim_x)
          + global_norm_idx(ripple::dim_y);
    }
  );
  g.apply_pipeline(pipeline);

  for (auto j : ripple::range(g.size(ripple::dim_y))) {
    for (auto i : ripple::range(g.size(ripple::dim_x))) {
      const auto norm_idx_x = static_cast<real_t>(i) / size_x;
      const auto norm_idx_y = static_cast<real_t>(j) / size_y;
      EXPECT_NEAR(*g(i, j), norm_idx_x + norm_idx_y, 1e-7);
    }
  }
}

TEST(container_grid, can_get_normalized_global_index_3d) {
  constexpr auto size_x = size_t{107};
  constexpr auto size_y = size_t{64};
  constexpr auto size_z = size_t{73};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g(topo, size_x, size_y, size_z);
    
  auto pipeline = ripple::make_pipeline(
    [] ripple_host_device (auto it) {
      *it = ripple::global_norm_idx(ripple::dim_x)
          + ripple::global_norm_idx(ripple::dim_y)
          + ripple::global_norm_idx(ripple::dim_z);
    }
  );
  g.apply_pipeline(pipeline);

  for (auto k : ripple::range(g.size(ripple::dim_z))) {
    for (auto j : ripple::range(g.size(ripple::dim_y))) {
      for (auto i : ripple::range(g.size(ripple::dim_x))) {
        const auto norm_idx_x = static_cast<real_t>(i) / size_x;
        const auto norm_idx_y = static_cast<real_t>(j) / size_y;
        const auto norm_idx_z = static_cast<real_t>(k) / size_z;
        EXPECT_NEAR(*g(i, j, k), norm_idx_x + norm_idx_y + norm_idx_z, 1e-7);
      }
    }
  }
}

#endif // RIPPLE_TESTS_CONTAINER_DEVICE_GRID_TESTS_HPP

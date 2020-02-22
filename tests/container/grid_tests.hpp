//==--- ripple/core/tests/container/grid_tests.hpp -------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  grid_tests.hpp
/// \brief This file defines tests for grid functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_GRID_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_GRID_TESTS_HPP

#include <ripple/core/container/grid.hpp>

// Defines the data type for the tests.
using real_t = float;

TEST(container_grid, can_create_grid_1d) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, size_x);

  EXPECT_EQ(g.size()             , size_t{size_x});
  EXPECT_EQ(g.size(ripple::dim_x), size_t{size_x});
}

TEST(container_grid, can_create_grid_2d) {
  constexpr auto size_x = size_t{10};
  constexpr auto size_y = size_t{10};
  auto topo = ripple::Topology();
  ripple::grid_2d_t<real_t> g(topo, size_x, size_y);

  EXPECT_EQ(g.size()             , size_t{size_x * size_y});
  EXPECT_EQ(g.size(ripple::dim_x), size_t{size_x});
  EXPECT_EQ(g.size(ripple::dim_y), size_t{size_y});
}

TEST(container_grid, can_create_grid_3d) {
  constexpr auto size_x = size_t{10};
  constexpr auto size_y = size_t{10};
  constexpr auto size_z = size_t{5};
  auto topo = ripple::Topology();
  ripple::grid_3d_t<real_t> g(topo, size_x, size_y, size_z);

  EXPECT_EQ(g.size(), size_t{size_x * size_y * size_z});
  EXPECT_EQ(g.size(ripple::dim_x), size_t{size_x});
  EXPECT_EQ(g.size(ripple::dim_y), size_t{size_y});
  EXPECT_EQ(g.size(ripple::dim_z), size_t{size_z});
}

//==--- [access] -----------------------------------------------------------==//

TEST(container_grid, can_access_simple_elements_1d) {
  constexpr auto size_x = size_t{100};
  auto topo = ripple::Topology();
  ripple::grid_1d_t<real_t> g(topo, size_x);
      
  for (auto i : ripple::range(g.size())) {
    *g(i) = static_cast<float>(i) + 5.0f;
  };
  for (auto i : ripple::range(g.size())) {
    EXPECT_EQ(*g(i), static_cast<float>(i) + 5.0f);
  }
}

#endif // RIPPLE_TESTS_CONTAINER_GRID_TESTS_HPP

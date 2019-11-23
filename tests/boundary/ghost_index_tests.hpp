//==--- ripple/tests/boundary/ghost_index_tests.hpp -------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  ghost_index_tests.hpp
/// \brief This file defines tests for ghost indices.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_BOUNDARY_GHOST_INDEX_TESTS_HPP
#define RIPPLE_TESTS_BOUNDARY_GHOST_INDEX_TESTS_HPP

#include <ripple/boundary/ghost_index.hpp>
#include <gtest/gtest.h>

TEST(boundary_ghost_index, can_determine_location) {
  ripple::ghost_index_3d_t indices;
  indices.index(ripple::dim_x) = 1;
  indices.index(ripple::dim_y) = 2;
  indices.index(ripple::dim_z) = 3;

  for (auto i : ripple::range(indices.dimensions())) {
    EXPECT_TRUE(indices.is_front(i));
    EXPECT_FALSE(indices.is_back(i));
    EXPECT_FALSE(indices.is_void(i));
  }

  indices.index(ripple::dim_x) = -1;
  indices.index(ripple::dim_y) = -2;
  indices.index(ripple::dim_z) = -3;

  for (auto i : ripple::range(indices.dimensions())) {
    EXPECT_FALSE(indices.is_front(i));
    EXPECT_TRUE(indices.is_back(i));
    EXPECT_FALSE(indices.is_void(i));
  }
}

TEST(boundary_ghost_index, can_set_as_void) {
  ripple::ghost_index_3d_t indices;

  for (auto i : ripple::range(indices.dimensions())) {
    indices.set_as_void(i);
    EXPECT_TRUE(indices.is_void(i));
  }

  indices.index(ripple::dim_x) = -1;
  indices.index(ripple::dim_y) = -2;
  indices.index(ripple::dim_z) = -3;

  for (auto i : ripple::range(indices.dimensions())) {
    EXPECT_FALSE(indices.is_front(i));
    EXPECT_TRUE(indices.is_back(i));
    EXPECT_FALSE(indices.is_void(i));
  }
}

TEST(boundary_ghost_index, can_get_dimensions) {
  ripple::ghost_index_3d_t indices;
  EXPECT_EQ(indices.dimensions(), std::size_t{3});
}

#endif // RIPPLE_TESTS_BOUNDARY_GHOST_INDEX_TESTS_HPP

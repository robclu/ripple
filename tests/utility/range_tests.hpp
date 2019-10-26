//==--- ripple/tests/utility/range_tests.hpp --------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  range_tests.hpp
/// \brief This file implements tests for the range functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_UTILITY_RANGE_TESTS_HPP
#define RIPPLE_TESTS_UTILITY_RANGE_TESTS_HPP

#include <ripple/utility/range.hpp>
#include <gtest/gtest.h>

TEST(utility_range, can_create_simple_range) {
  int i = 0;
  for (auto r : ripple::range(int{100})) {
    EXPECT_EQ(r, i++);
  }
}

TEST(utility_range, can_create_stepped_range) {
  int i = 10, end = 100, step = 2;
  for (auto r : ripple::range(i, end, step)) {
    EXPECT_EQ(r, i);
    i += step;
  }
}

TEST(utility_range, range_works_with_non_integer_types) {
  float i = 0.3f, end = 0.9f, step = 0.1f;
  for (auto f : ripple::range(i, end, step)) {
    EXPECT_EQ(f, i);
    i += step;
  }
}

#endif // RIPPLE_TESTS_UTILITY_RANGE_TESTS_HPP

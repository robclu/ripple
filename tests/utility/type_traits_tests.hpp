//==--- ripple/tests/utility/type_traits_tests.hpp --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  typpe_traits_tests.hpp
/// \brief This file implements tests for the range functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_UTILITY_TYPE_TRAITS_TESTS_HPP
#define RIPPLE_TESTS_UTILITY_TYPE_TRAITS_TESTS_HPP

#include <ripple/utility/type_traits.hpp>
#include <gtest/gtest.h>

TEST(utility_traits, all_arithmetic_v) {
  const auto b1 = ripple::all_arithmetic_v<int, float, double, std::size_t>;
  const auto b2 = ripple::all_arithmetic_v<int*, float, double>;
  const auto b3 = ripple::all_arithmetic_v<int, float, double*>;
  EXPECT_TRUE(b1);
  EXPECT_FALSE(b2);
  EXPECT_FALSE(b3);
}

TEST(utility_traits, all_same_v) {
  const auto b1 = ripple::all_same_v<int, float, double, std::size_t>;
  const auto b2 = ripple::all_same_v<int, int, int>;
  const auto b3 = ripple::all_same_v<int, int*, int>;

  EXPECT_FALSE(b1);
  EXPECT_TRUE(b2);
  EXPECT_FALSE(b3);
}

#endif // RIPPLE_TESTS_UTILITY_TYPE_TRAITS_TESTS_HPP

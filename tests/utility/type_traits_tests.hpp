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

TEST(utility_type_traits, all_arithmetic) {
  const auto b1 = ripple::all_arithmetic_v<int, float, double, std::size_t>;
  const auto b2 = ripple::all_arithmetic_v<int*, float, double>;
  const auto b3 = ripple::all_arithmetic_v<int, float, double*>;
  EXPECT_TRUE(b1);
  EXPECT_FALSE(b2);
  EXPECT_FALSE(b3);
}

TEST(utility_type_traits, all_same) {
  const auto b1 = ripple::all_same_v<int, float, double, std::size_t>;
  const auto b2 = ripple::all_same_v<int, int, int>;
  const auto b3 = ripple::all_same_v<int, int*, int>;

  EXPECT_FALSE(b1);
  EXPECT_TRUE(b2);
  EXPECT_FALSE(b3);
}

TEST(utility_type_traits, any_arithmetic) {
  const auto b1 = ripple::any_arithmetic_v<int, float, double, std::size_t>;
  const auto b2 = ripple::any_arithmetic_v<int*, float*, double*>;
  const auto b3 = ripple::any_arithmetic_v<int, float, double*>;
  EXPECT_TRUE(b1);
  EXPECT_FALSE(b2);
  EXPECT_TRUE(b3);
}

TEST(utility_type_traits, any_same) {
  const auto b1 = ripple::any_same_v<int, float, double, std::size_t, int>;
  const auto b2 = ripple::any_same_v<int, int, int>;
  const auto b3 = ripple::any_same_v<int, int*, float>;

  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);
  EXPECT_FALSE(b3);
}

enum class Value : int { v0, v1 };
template <typename T> struct W {};
template <Value v> struct X {};
template <int...> struct Y {};
template <typename T, typename... Ts> struct Z {};

TEST(utility_type_traits, is_same_ignoring_templates) {
  auto b1 = ripple::is_same_ignoring_templates_v<int, int>;
  auto b2 = ripple::is_same_ignoring_templates_v<int, float>;
  EXPECT_TRUE(b1);
  EXPECT_FALSE(b2);

  b1 = ripple::is_same_ignoring_templates_v<int, int&>;
  b2 = ripple::is_same_ignoring_templates_v<int, std::decay_t<int&>>;
  EXPECT_FALSE(b1);
  EXPECT_TRUE(b2);

  // More complicated tests:
  b1 = ripple::is_same_ignoring_templates_v<W<int>, W<float>>;
  EXPECT_TRUE(b1);

  b1 = ripple::is_same_ignoring_templates_v<X<Value::v0>, X<Value::v1>>;
  b2 = ripple::is_same_ignoring_templates_v<X<Value::v0>, X<Value::v0>>;
  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);

  b1 = ripple::is_same_ignoring_templates_v<Y<4, 3>, Y<4>>;
  b2 = ripple::is_same_ignoring_templates_v<Y<1,2 , 3>, Y<3, 2, 1>>;
  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);

  b1 = ripple::is_same_ignoring_templates_v<Y<4, 3>, X<Value::v0>>;
  b2 = ripple::is_same_ignoring_templates_v<Y<1, 2, 3>, Z<int, float>>;
  EXPECT_FALSE(b1);
  EXPECT_FALSE(b2);

  b1 = ripple::is_same_ignoring_templates_v<Z<int, float>, Z<double>>;
  b2 = ripple::is_same_ignoring_templates_v<Z<double>, Z<double>>;
  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);
}

#endif // RIPPLE_TESTS_UTILITY_TYPE_TRAITS_TESTS_HPP

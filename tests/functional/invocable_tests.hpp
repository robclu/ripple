//==--- ripple/tests/functional/invocable_tests.hpp -------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invocable_tests.hpp
/// \brief This file defines tests for invocable functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_FUNCTIONAL_INVOCABLE_TESTS_HPP
#define RIPPLE_TESTS_FUNCTIONAL_INVOCABLE_TESTS_HPP

#include <ripple/functional/invocable.hpp>
#include <gtest/gtest.h>

TEST(functional_invocable, can_store_lambda_and_invoke_it) {
  auto l = [] (int x, float y) {
    EXPECT_EQ(x, 7);
    EXPECT_EQ(y, 1.0f);
  };

  int   x = 7;
  float y = 1.0f;

  auto invocable = ripple::make_invocable(l, x, y);

  invocable();

  x = 8;

  // Test that the invocable makes a copy.
  invocable();
}

TEST(functional_invocable, can_store_auto_lambda_and_invoke_it) {
  auto l = [] (auto x, auto y) {
    EXPECT_EQ(x, 7);
    EXPECT_EQ(y, 1.0f);
  };

  int   x = 7;
  float y = 1.0f;

  auto invocable = ripple::make_invocable(l);

  invocable(x, y);
}

TEST(functional_invocable, can_store_mixed_lambda_and_invoke_it) {
  auto l = [] (auto x, auto y, int z) {
    EXPECT_EQ(x, 7);
    EXPECT_EQ(y, 1.0f);
    EXPECT_EQ(z, 12);
  };

  int   x = 7;
  float y = 1.0f;

  auto invocable = ripple::make_invocable(l);

  invocable(x, y, 12);

  // Store the known parameter:
  auto oth_invocable = ripple::make_invocable(l, 12);

  oth_invocable(x, y);
}

TEST(functional_invocable, can_determine_if_type_is_invocable) {
  auto l = [] (auto x, auto y) {
    EXPECT_EQ(x, 7);
    EXPECT_EQ(y, 1.0f);
  };

  using l_conv_t = ripple::make_invocable_t<decltype(l)>;

  auto invocable = ripple::make_invocable(l);
  auto l_conv    = l_conv_t{l};

  constexpr auto l_inv_v  = ripple::is_invocable_v<decltype(l)>;
  constexpr auto i_inv_v  = ripple::is_invocable_v<decltype(invocable)>;
  constexpr auto lc_inv_v = ripple::is_invocable_v<decltype(l_conv)>;

  EXPECT_FALSE(l_inv_v);
  EXPECT_TRUE(i_inv_v);
  EXPECT_TRUE(lc_inv_v);
}

#endif // RIPPLE_TESTS_FUNCTIONAL_INVOCABLE_TESTS_HPP



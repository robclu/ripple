//==--- ripple/core/tests/math/math_tests.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  math_tests.hpp
/// \brief This file defines tests for math functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_MATH_MATH_TESTS_HPP
#define RIPPLE_TESTS_MATH_MATH_TESTS_HPP

#include "lerp_tests.hpp"
#include <ripple/core/container/vec.hpp>
#include <ripple/core/math/math.hpp>
#include <gtest/gtest.h>

constexpr double tol = 1e-6;

TEST(math_tests, sign_signed_int) {
  const int a = 10;
  const int b = -22;
  const int c = 0;
  EXPECT_EQ(ripple::math::sign(a), 1 );
  EXPECT_EQ(ripple::math::sign(b), -1);
  EXPECT_EQ(ripple::math::sign(c), 0 );
}

TEST(math_tests, sign_unsigned_int) {
  const std::size_t a = 10;
  const std::size_t b = 0;
  EXPECT_EQ(ripple::math::sign(a), std::size_t{1});
  EXPECT_EQ(ripple::math::sign(b), std::size_t{0});
}

TEST(math_tests, sign_float) {
  const float a = 10.0f;
  const float b = -22.0f;
  const float c = 0.0f;
  EXPECT_EQ(ripple::math::sign(a), 1.0f);
  EXPECT_EQ(ripple::math::sign(b), -1.0f);
  EXPECT_EQ(ripple::math::sign(c), 0.0f );
}

TEST(math_tests, sign_double) {
  const double a = 10.0;
  const double b = -22.0;
  const double c = 0.0;
  EXPECT_EQ(ripple::math::sign(a), 1.0);
  EXPECT_EQ(ripple::math::sign(b), -1.0);
  EXPECT_EQ(ripple::math::sign(c), 0.0);
}

TEST(math_tests, sqrt_non_vec_type) {
  EXPECT_NEAR(ripple::math::sqrt(4), std::sqrt(4), tol);
}

TEST(math_tests, sqrt_vec_type_owned) {
  using namespace ripple;
  using type_t = Vector<float, 3, contiguous_owned_t>;

  type_t x(2, 3, 4);
  auto t = math::sqrt(x);

  EXPECT_NEAR(t[0], std::sqrt(2), tol);
  EXPECT_NEAR(t[1], std::sqrt(3), tol);
  EXPECT_NEAR(t[2], std::sqrt(4), tol);
}

#endif // RIPPLE_TESTS_MATH_TESTS_HPP


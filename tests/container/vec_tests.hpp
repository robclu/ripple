//==--- ripple/core/tests/container/vec_tests.cpp --------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vec_tests.hpp
/// \brief This file contains tests for vec.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_VEC_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_VEC_TESTS_HPP

#include <ripple/core/container/vec.hpp>
#include <gtest/gtest.h>

TEST(container_vec, can_create_vec_default_constructor) {
  auto v = ripple::Vec<float, 3>();
  EXPECT_TRUE(v.size() == 3);
}

TEST(container_vec, can_create_vec_with_values_and_modify_them) {
  auto v = ripple::Vec<int, 3>(1, 2, 3);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);

  v[0] = 10; v[1] = 20; v[2] = 30;

  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
}

TEST(container_vec, can_create_set_and_modify_vec_values) {
  auto v = ripple::Vec<int, 3>();
  v[0] = 1; v[1] = 2; v[2] = 3;
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
}

TEST(container_vec, can_add_vecs) {
  auto v = ripple::Vec<int, 3>(1, 2, 3);
  v += v;
  EXPECT_EQ(v[0], 2);
  EXPECT_EQ(v[1], 4);
  EXPECT_EQ(v[2], 6);
}

TEST(container_vec, can_add_vecs_and_scalar) {
  auto v = ripple::Vec<int, 3>(1, 2, 3);
  v += 5;
  EXPECT_EQ(v[0], 6);
  EXPECT_EQ(v[1], 7);
  EXPECT_EQ(v[2], 8);

  auto v2 = v + 4;
  EXPECT_EQ(v2[0], 10);
  EXPECT_EQ(v2[1], 11);
  EXPECT_EQ(v2[2], 12);

  auto v3 = 7 + v2;
  EXPECT_EQ(v3[0], 17);
  EXPECT_EQ(v3[1], 18);
  EXPECT_EQ(v3[2], 19);
}

TEST(container_vec, can_subtract_vecs) {
  auto v = ripple::Vec<int, 3>(1, 2, 3);
  v -= (v + v);
  EXPECT_EQ(v[0], -1);
  EXPECT_EQ(v[1], -2);
  EXPECT_EQ(v[2], -3);
}

TEST(container_vec, can_subtract_vecs_and_scalar) {
  auto v = ripple::Vec<int, 3>(11, 12, 13);
  v -= 5;
  EXPECT_EQ(v[0], 6);
  EXPECT_EQ(v[1], 7);
  EXPECT_EQ(v[2], 8);

  auto v2 = v - 1;
  EXPECT_EQ(v2[0], 5);
  EXPECT_EQ(v2[1], 6);
  EXPECT_EQ(v2[2], 7);

  auto v3 = 7 - v2;
  EXPECT_EQ(v3[0], 2);
  EXPECT_EQ(v3[1], 1);
  EXPECT_EQ(v3[2], 0);
}

TEST(container_vec, can_multiply_vecs) {
  auto v = ripple::Vec<int, 3>(1, 2, 3);
  v *= v;
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 4);
  EXPECT_EQ(v[2], 9);
}

TEST(container_vec, can_multiply_vecs_and_scalar) {
  auto v = ripple::Vec<int, 3>(1, 2, 3);
  v *= 5;
  EXPECT_EQ(v[0], 5);
  EXPECT_EQ(v[1], 10);
  EXPECT_EQ(v[2], 15);

  auto v2 = v *  2 * v;
  EXPECT_EQ(v2[0], 50);
  EXPECT_EQ(v2[1], 200);
  EXPECT_EQ(v2[2], 450);

  auto v3 = 10 * v;
  EXPECT_EQ(v3[0], 50);
  EXPECT_EQ(v3[1], 100);
  EXPECT_EQ(v3[2], 150);
}

TEST(container_vec, can_divide_vecs) {
  auto u = ripple::Vec<int, 3>(2, 3, 4);
  auto v = ripple::Vec<int, 3>(12, 21, 36);
  v /= u;
  EXPECT_EQ(v[0], 6);
  EXPECT_EQ(v[1], 7);
  EXPECT_EQ(v[2], 9);
}

TEST(container_vec, can_divide_vecs_and_scalar) {
  auto v = ripple::Vec<int, 3>(10, 20, 30);
  v /= 5;
  EXPECT_EQ(v[0], 2);
  EXPECT_EQ(v[1], 4);
  EXPECT_EQ(v[2], 6);

  auto v2 = 12 / v;
  EXPECT_EQ(v2[0], 6);
  EXPECT_EQ(v2[1], 3);
  EXPECT_EQ(v2[2], 2);

  auto v3 = 10 * v / 5;
  EXPECT_EQ(v3[0], 4);
  EXPECT_EQ(v3[1], 8);
  EXPECT_EQ(v3[2], 12);
}

#endif // RIPPLE_TESTS_CONTAINER_VEC_TESTS_HPP

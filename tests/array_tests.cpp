//==--- cpp/tests/array_tests.cpp -------------------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Streamline.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array_tests.cpp
/// \brief This file contains tests for arrays.
//
//==------------------------------------------------------------------------==//

#include <streamline/container/vec.hpp>
#include <gtest/gtest.h>

TEST(vec, can_create_vec_default_constructor) {
  auto v = streamline::Vec<float, 3>();
  EXPECT_TRUE(v.size() == 3);
}

TEST(vec, can_create_vec_with_values_and_modify_them) {
  auto v = streamline::Vec<int, 3>(1, 2, 3);
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);

  v[0] = 10; v[1] = 20; v[2] = 30;

  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
}

TEST(vec, can_create_set_and_modify_vec_values) {
  auto v = streamline::Vec<int, 3>();
  v[0] = 1; v[1] = 2; v[2] = 3;
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);
}

TEST(vec, can_add_vecs) {
  auto v = streamline::Vec<int, 3>(1, 2, 3);
  v += v;
  EXPECT_EQ(v[0], 2);
  EXPECT_EQ(v[1], 4);
  EXPECT_EQ(v[2], 6);
}

TEST(vec, can_add_vecs_and_scalar) {
  auto v = streamline::Vec<int, 3>(1, 2, 3);
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

TEST(vec, can_subtract_vecs) {
  auto v = streamline::Vec<int, 3>(1, 2, 3);
  v -= (v + v);
  EXPECT_EQ(v[0], -1);
  EXPECT_EQ(v[1], -2);
  EXPECT_EQ(v[2], -3);
}

TEST(vec, can_subtract_vecs_and_scalar) {
  auto v = streamline::Vec<int, 3>(11, 12, 13);
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

TEST(vec, can_multiply_vecs) {
  auto v = streamline::Vec<int, 3>(1, 2, 3);
  v *= v;
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 4);
  EXPECT_EQ(v[2], 9);
}

TEST(vec, can_multiply_vecs_and_scalar) {
  auto v = streamline::Vec<int, 3>(1, 2, 3);
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

TEST(vec, can_divide_vecs) {
  auto u = streamline::Vec<int, 3>(2, 3, 4);
  auto v = streamline::Vec<int, 3>(12, 21, 36);
  v /= u;
  EXPECT_EQ(v[0], 6);
  EXPECT_EQ(v[1], 7);
  EXPECT_EQ(v[2], 9);
}

TEST(vec, can_divide_vecs_and_scalar) {
  auto v = streamline::Vec<int, 3>(10, 20, 30);
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

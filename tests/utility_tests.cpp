//==--- streamline/tests/utility_tests.cpp ----------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  utility_tests.cpp
/// \brief This file implements tests for utility functionality.
//
//==------------------------------------------------------------------------==//

#include <streamline/utility/range.hpp>
#include <streamline/utility/number.hpp>
#include <gtest/gtest.h>

//==--- [int64] ------------------------------------------------------------==//

TEST(int64, can_create_constexpr_int64) {
  constexpr auto num = streamline::Int64<1000>();
  EXPECT_EQ(int64_t{1000}, static_cast<int64_t>(num));
}

TEST(int64, can_create_and_covert_to_value) {
  auto num = streamline::Int64<2000>();
  EXPECT_EQ(int64_t{2000}, static_cast<int64_t>(num));
}

//==--- [number] -----------------------------------------------------------==//

TEST(number, can_create_constexpr_number) {
  constexpr auto num = streamline::Num<20>();
  EXPECT_EQ(std::size_t{20}, static_cast<std::size_t>(num));
}

TEST(number, can_create_and_covert_to_value) {
  auto num = streamline::Num<20>();
  EXPECT_EQ(std::size_t{20}, static_cast<std::size_t>(num));
}

//==--- [range] ------------------------------------------------------------==//

TEST(range, can_create_simple_range) {
  int i = 0;
  for (auto r : streamline::range(int{100})) {
    EXPECT_EQ(r, i++);
  }
}

TEST(range, can_create_stepped_range) {
  int i = 10, end = 100, step = 2;
  for (auto r : streamline::range(i, end, step)) {
    EXPECT_EQ(r, i);
    i += step;
  }
}

TEST(range, range_works_with_non_integer_types) {
  float i = 0.3f, end = 0.9f, step = 0.1f;
  for (auto f : streamline::range(i, end, step)) {
    EXPECT_EQ(f, i);
    i += step;
  }
}

//==--- [traits] -----------------------------------------------------------==//

TEST(traits, all_arithmetic_v) {
  const auto b1 = streamline::all_arithmetic_v<int, float, double, std::size_t>;
  const auto b2 = streamline::all_arithmetic_v<int*, float, double>;
  const auto b3 = streamline::all_arithmetic_v<int, float, double*>;
  EXPECT_TRUE(b1);
  EXPECT_FALSE(b2);
  EXPECT_FALSE(b3);
}

TEST(traits, all_same_v) {
  const auto b1 = streamline::all_same_v<int, float, double, std::size_t>;
  const auto b2 = streamline::all_same_v<int, int, int>;
  const auto b3 = streamline::all_same_v<int, int*, int>;

  EXPECT_FALSE(b1);
  EXPECT_TRUE(b2);
  EXPECT_FALSE(b3);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

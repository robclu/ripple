//==--- streamline/tests/algorithm_tests.cpp --------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  algorithm_tests.cpp
/// \brief This file defines tests for the algorithm functionality.
//
//==------------------------------------------------------------------------==//

#include <streamline/algorithm/unrolled_for.hpp>
#include <gtest/gtest.h>

//==--- [compile time unrolling] -------------------------------------------==//

TEST(algorithm, can_compile_time_unroll) {
  constexpr auto amount = std::size_t{3};
  auto sum = std::size_t{0};
  streamline::unrolled_for<amount>([&sum] (auto i) {
    sum += i;
  });
  EXPECT_EQ(sum, amount);
}

TEST(algorithm, can_compile_time_unroll_above_max_unroll_depth) {
  constexpr auto amount = std::size_t{64};
  static_assert(
    amount > streamline_max_unroll_depth,            
    "Increase the amount of the max unrolling depth!"
  );

  int sum = 0, result = 0;
  streamline::unrolled_for_bounded<amount>([&sum] (auto i) {
    sum += i;
  });

  for (const auto i : streamline::range(amount)) {
    result += static_cast<int>(i);
  }
  EXPECT_EQ(sum, result);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

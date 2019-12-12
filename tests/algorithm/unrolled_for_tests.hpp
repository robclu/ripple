//==--- ripple/tests/algorithm/unrolled_for_tests.hpp ------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  unrolled_for_tests.hpp
/// \brief This file defines tests for unrolled_for.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_ALGORITHM_UNROLLED_FOR_TESTS_HPP
#define RIPPLE_TESTS_ALGORITHM_UNROLLED_FOR_TESTS_HPP

#include <ripple/algorithm/unrolled_for.hpp>
#include <gtest/gtest.h>

//==--- [compile time unrolling] -------------------------------------------==//

TEST(algorithm_unrolled_for, can_compile_time_unroll) {
  constexpr auto amount = std::size_t{3};
  auto sum = std::size_t{0};
  ripple::unrolled_for<amount>([&sum] (auto i) {
    sum += i;
  });
  EXPECT_EQ(sum, amount);
}

TEST(algorithm_unrolled_for, can_compile_time_unroll_above_max_unroll_depth) {
  constexpr auto amount = std::size_t{64};
  static_assert(
    amount > ripple_max_unroll_depth,            
    "Increase the amount of the max unrolling depth!"
  );

  int sum = 0, result = 0;
  ripple::unrolled_for_bounded<amount>([&sum] (auto i) {
    sum += i;
  });

  for (const auto i : ripple::range(amount)) {
    result += static_cast<int>(i);
  }
  EXPECT_EQ(sum, result);
}

#endif // RIPPLE_TESTS_ALGORITHM_UNROLLED_FOR_TESTS_HPP


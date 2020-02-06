//==--- ripple/core/tests/utility/number_tests.hpp -------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  number_tests.hpp
/// \brief This file implements tests for compile time number functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_UTILITY_NUMBER_TESTS_HPP
#define RIPPLE_TESTS_UTILITY_NUMBER_TESTS_HPP

#include <ripple/core/utility/number.hpp>
#include <gtest/gtest.h>

//==--- [int64] ------------------------------------------------------------==//

TEST(utility_int64, can_create_constexpr_int64) {
  constexpr auto num = ripple::Int64<1000>();
  EXPECT_EQ(int64_t{1000}, static_cast<int64_t>(num));
}

TEST(utility_int64, can_create_and_covert_to_value) {
  auto num = ripple::Int64<2000>();
  EXPECT_EQ(int64_t{2000}, static_cast<int64_t>(num));
}

//==--- [number] -----------------------------------------------------------==//

TEST(utility_number, can_create_constexpr_number) {
  constexpr auto num = ripple::Num<20>();
  EXPECT_EQ(std::size_t{20}, static_cast<std::size_t>(num));
}

TEST(utility_number, can_create_and_covert_to_value) {
  auto num = ripple::Num<20>();
  EXPECT_EQ(std::size_t{20}, static_cast<std::size_t>(num));
}

#endif // RIPPLE_TESTS_UTILITY_NUMBER_TESTS_HPP


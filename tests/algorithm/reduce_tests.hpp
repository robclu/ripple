//==--- ripple/tests/algorithm/reduce_tests.hpp ------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce_tests.hpp
/// \brief This file defines tests for reductions.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_ALGORITHM_REDUCE_FOR_TESTS_HPP
#define RIPPLE_TESTS_ALGORITHM_REDUCE_FOR_TESTS_HPP

#include <ripple/algorithm/reduce.hpp>
#include <gtest/gtest.h>

// Functor to define a sum operation for reduction.
struct SumReducer {
  template <typename T>
  auto operator()(T& into, T& from) -> void {
    into += from;
  }
};

TEST(algorithm_reduce, can_sum_reduce_1d) {
  using data_t = float;
  constexpr auto size_x = size_t{101};
  ripple::host_block_1d_t<data_t> block(size_x);

  ripple::invoke(block, [] (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, SumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x);
}

TEST(algorithm_reduce, can_sum_reduce_with_padding_1d) {
  using data_t = float;
  constexpr auto size_x  = size_t{103};
  constexpr auto padding = 2;
  ripple::host_block_1d_t<data_t> block(padding, size_x);

  ripple::invoke(block, [] (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, SumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x);
}

TEST(algorithm_reduce, can_sum_reduce_2d) {
  using data_t = float;
  constexpr auto size_x = size_t{131};
  constexpr auto size_y = size_t{112};
  ripple::host_block_2d_t<data_t> block(size_x, size_y);

  ripple::invoke(block, [] (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, SumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y);
}

TEST(algorithm_reduce, can_sum_reduce_with_padding_2d) {
  using data_t = float;
  constexpr auto size_x  = size_t{131};
  constexpr auto size_y  = size_t{112};
  constexpr auto padding = 3;
  ripple::host_block_2d_t<data_t> block(padding, size_x, size_y);

  ripple::invoke(block, [] (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, SumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y);
}

TEST(algorithm_reduce, can_sum_reduce_3d) {
  using data_t = float;
  constexpr auto size_x = size_t{131};
  constexpr auto size_y = size_t{112};
  constexpr auto size_z = size_t{11};
  ripple::host_block_3d_t<data_t> block(size_x, size_y, size_z);

  ripple::invoke(block, [] (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, SumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y * size_z);
}

TEST(algorithm_reduce, can_sum_reduce_with_padding_3d) {
  using data_t = float;
  constexpr auto size_x  = size_t{131};
  constexpr auto size_y  = size_t{112};
  constexpr auto size_z  = size_t{9};
  constexpr auto padding = 3;
  ripple::host_block_3d_t<data_t> block(padding, size_x, size_y, size_z);

  ripple::invoke(block, [] (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, SumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y * size_z);
}

#endif // RIPPLE_TESTS_ALGORITHM_UNROLLED_FOR_TESTS_HPP

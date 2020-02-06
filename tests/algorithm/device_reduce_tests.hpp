//==--- ripple/core/tests/algorithm/device_reduce_tests.hpp ----- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_reduce_tests.hpp
/// \brief This file defines tests for reductions on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_ALGORITHM_DEVICE_REDUCE_TESTS_HPP
#define RIPPLE_TESTS_ALGORITHM_DEVICE_REDUCE_TESTS_HPP

#include <ripple/core/algorithm/reduce.hpp>
#include <gtest/gtest.h>

// Functor to define a sum operation for reduction.
struct DeviceSumReducer {
  template <typename T>
  ripple_host_device auto operator()(T& into, T& from) -> void {
    *into += *from;
  }
};

TEST(algorithm_device_reduce, can_sum_reduce_1d) {
  using data_t = float;
  constexpr auto size_x = size_t{101};
  ripple::device_block_1d_t<data_t> block(size_x);

  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, DeviceSumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x);
}

TEST(algorithm_device_reduce, can_sum_reduce_with_padding_1d) {
  using data_t = float;
  constexpr auto size_x  = size_t{103};
  constexpr auto padding = 2;
  ripple::device_block_1d_t<data_t> block(padding, size_x);

  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, DeviceSumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x);
}

TEST(algorithm_device_reduce, can_sum_reduce_2d) {
  using data_t = float;
  constexpr auto size_x = size_t{201};
  constexpr auto size_y = size_t{19};
  ripple::device_block_2d_t<data_t> block(size_x, size_y);

  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, DeviceSumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y);
}

TEST(algorithm_device_reduce, can_sum_reduce_with_padding_2d) {
  using data_t = float;
  constexpr auto size_x  = size_t{131};
  constexpr auto size_y  = size_t{119};
  constexpr auto padding = 3;
  ripple::device_block_2d_t<data_t> block(padding, size_x, size_y);

  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, DeviceSumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y);
}

TEST(algorithm_device_reduce, can_sum_reduce_3d) {
  using data_t = float;
  constexpr auto size_x = size_t{207};
  constexpr auto size_y = size_t{19};
  constexpr auto size_z = size_t{271};
  ripple::device_block_3d_t<data_t> block(size_x, size_y, size_z);

  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, DeviceSumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y * size_z);
}

TEST(algorithm_device_reduce, can_sum_reduce_with_padding_3d) {
  using data_t = float;
  constexpr auto size_x  = size_t{297};
  constexpr auto size_y  = size_t{13};
  constexpr auto size_z  = size_t{117};
  constexpr auto padding = 3;
  ripple::device_block_3d_t<data_t> block(padding, size_x, size_y, size_z);

  ripple::invoke(block, [] ripple_host_device (auto it) {
    *it = data_t{1};
  });

  auto result = ripple::reduce(block, DeviceSumReducer());
  EXPECT_EQ(static_cast<size_t>(result), size_x * size_y * size_z);
}

#endif // RIPPLE_TESTS_ALGORITHM_DEVICE_REDUCE_TESTS_HPP

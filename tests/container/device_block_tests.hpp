//==--- ripple/tests/container/device_block_tests.hpp ------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  device_block_tests.hpp
/// \brief This file defines tests for device block functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_DEVICE_BLOCK_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_DEVICE_BLOCK_TESTS_HPP

#include <ripple/container/block_traits.hpp>
#include <ripple/container/device_block.hpp>
#include <ripple/storage/storage_descriptor.hpp>
#include <ripple/storage/stridable_layout.hpp>
#include <ripple/utility/dim.hpp>
#include <gtest/gtest.h>

TEST(container_device_block, can_create_block_1d) {
  ripple::device_block_1d_t<float> b(20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20));
}

TEST(container_device_block, can_create_block_2d) {
  ripple::device_block_2d_t<int> b(20, 20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20 * 20));
}

TEST(container_device_block, can_create_block_3d) {
  ripple::device_block_3d_t<double> b(20, 20, 20);
  EXPECT_EQ(b.size(), static_cast<decltype(b.size())>(20 * 20 * 20));
}

#endif // RIPPLE_TESTS_CONTAINER_DEVICE_BLOCK_TESTS_HPP



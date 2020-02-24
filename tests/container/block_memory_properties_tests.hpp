//==--- ../tests/container/block_memory_properties_tests.hpp -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block_memory_properties_tests.hpp
/// \brief This file defines tests for block memory property functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_BLOCK_MEMORY_PROPERTIES_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_BLOCK_MEMORY_PROPERTIES_TESTS_HPP

#include <ripple/core/container/block_memory_properties.hpp>
#include <gtest/gtest.h>

TEST(container_block_memory_props, can_copy_memory_properties) {
  ripple::BlockMemoryProps props;

  props.allocated  = true;
  props.async_copy = true;
  props.must_free  = true;
  props.pinned     = true;

  auto other_props = props;

  EXPECT_EQ(other_props.allocated , true);
  EXPECT_EQ(other_props.async_copy, true);
  EXPECT_EQ(other_props.must_free , true);
  EXPECT_EQ(other_props.pinned    , true);
}

TEST(container_block_memory_props, can_move_memory_properties) {
  ripple::BlockMemoryProps props;

  props.allocated  = true;
  props.async_copy = true;
  props.must_free  = true;
  props.pinned     = true;

  auto other_props = std::move(props);

  EXPECT_EQ(other_props.allocated , true);
  EXPECT_EQ(other_props.async_copy, true);
  EXPECT_EQ(other_props.must_free , true);
  EXPECT_EQ(other_props.pinned    , true);
}

#endif // RIPPLE_TESTS_CONTAINER_BLOCK_MEMORY_PROPERTIES_TESTS_HPP

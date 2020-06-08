//==--- tests/graph/node_tests.hpp ------------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  node_tests.hpp
/// \brief This file defines tests for node functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_GRAPH_NODE_TESTS_HPP
#define RIPPLE_TESTS_GRAPH_NODE_TESTS_HPP

#include <ripple/core/graph/node.hpp>
#include <gtest/gtest.h>

TEST(graph_node_executor, can_create_and_execute_no_args) {
  auto l = []() {
    EXPECT_TRUE(true);
  };

  ripple::NodeExecutorImpl<decltype(l)> exec(l);
  exec.execute();
}

TEST(graph_node_executor, can_create_and_execute_with_args) {
  auto l = [](int x, float y) {
    EXPECT_EQ(x, 7);
    EXPECT_EQ(y, 0.0f);
  };

  ripple::NodeExecutorImpl<decltype(l), int, float> exec(l, 7, 0.0f);
  exec.execute();
}

#endif // RIPPLE_TESTS_FUNCTIONAL_INVOCABLE_TESTS_HPP

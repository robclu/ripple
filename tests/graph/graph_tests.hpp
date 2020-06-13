//==--- tests/graph/graph_tests.hpp ------------------------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  graph_tests.hpp
/// \brief This file contains all graph tests.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_GRAPH_GRAPH_TESTS_HPP
#define RIPPLE_TESTS_GRAPH_GRAPH_TESTS_HPP

#include "node_tests.hpp"
#include <ripple/core/graph/graph.hpp>
#include <ripple/core/graph/graph_executor.hpp>

TEST(graph_tests, can_execute_parallel_graph) {
  using namespace ripple;
  int x = 0, y = 0, z = 0;
  // clang-format off
  auto graph = Graph();
  graph.emplace(
    Graph::make_node([&x] { x++; }),
    Graph::make_node([&y] { y++; }),
    Graph::make_node([&z] { z = 4; }));
  // clang-format on

  EXPECT_EQ(graph.size(), 3);
  graph_executor().execute(graph);
  graph_executor().wait_until_finished();

  EXPECT_EQ(x, 1);
  EXPECT_EQ(y, 1);
  EXPECT_EQ(z, 4);
}

TEST(graph_tests, can_execute_graph_with_dependency) {
  using namespace ripple;
  int x = 0, y = 0, z = 0;
  // clang-format off
  auto graph = Graph();
  graph.emplace(
    Graph::make_node([&x] { x++; }),
    Graph::make_node([&y] { y++; }),
    Graph::make_node([&z] { z = 4; }))
    .then([&x, &y, &z] {
      EXPECT_EQ(x, 1);
      EXPECT_EQ(y, 1);
      EXPECT_EQ(z, 4);
      x *= 2;
      y *= 3;
      z *= 2;
    })
    .then([&x, &y, &z] {
      EXPECT_EQ(x, 2);
      EXPECT_EQ(y, 3);
      EXPECT_EQ(z, 8);
      x++; y++; z++;
    });
  // clang-format on

  EXPECT_EQ(graph.size(), 5);
  graph_executor().execute(graph);
  graph_executor().wait_until_finished();

  EXPECT_EQ(x, 3);
  EXPECT_EQ(y, 4);
  EXPECT_EQ(z, 9);
}

#endif // RIPPLE_TESTS_GRAPH_GRAPH_TESTS_HPP
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

TEST(graph_graph, can_build_graph_node_by_node) {
  int x = 1;
  // clang-format off
  auto graph = ripple::Graph();
  graph
    .emplace([&x] { x++; })
    .emplace([&x] { x++; })
    .emplace([&x] { x *= 2; });
  // clang-format on

  EXPECT_EQ(graph.size(), 3);
}

TEST(graph_graph, can_build_graph_with_make_node) {
  using namespace ripple;
  int x = 1;
  // clang-format off
  auto graph = Graph();
  graph.emplace(
    Graph::make_node([&x] { x++; }),
    Graph::make_node([&x] { x++; }),
    Graph::make_node([&x] { x *= 2; }));
  // clang-format on

  EXPECT_EQ(graph.size(), 3);
}

#endif // RIPPLE_TESTS_GRAPH_GRAPH_TESTS_HPP
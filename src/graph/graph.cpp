/*==--- ripple/src/graph/graph.cpp ------------------------- -*- C++ -*- ---==**
 *
 *                                 Ripple
 *
 *               Copyright (c) 2019, 2020, 2021 Rob Clucas
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==------------------------------------------------------------------------==//
 *
 * \file  node.cpp
 * \brief This file implements a Node class for a graph.
 *
 *==------------------------------------------------------------------------==*/

#include <ripple/core/graph/graph.hpp>

namespace ripple {

auto Graph::set_allocation_pool_size(size_t nodes) noexcept -> bool {
  Guard g(initialization_lock());
  if (is_initialized()) {
    return false;
  }

  allocation_pool_nodes() = nodes;
  is_initialized()        = true;
  return true;
}

auto Graph::reset() noexcept -> void {
  for (auto& node : nodes_) {
    if (node) {
      info_allocator().recycle(node->info_);
      node_allocator().recycle(node);
    }
  }
  exec_count_ = 0;
}

auto Graph::clone() const noexcept -> Graph {
  Graph graph;
  for (const auto& node : nodes_) {
    // TODO: Add copying of node info for new node.
    graph.nodes_.emplace_back(node_allocator().create<NodeType>(*node));
    graph.nodes_.back()->info_ =
      info_allocator().create<NodeInfo>(node->info_->name, node->info_->id);
  }
  graph.join_ids_.clear();
  for (const auto& id : join_ids_) {
    graph.join_ids_.push_back(id);
  }
  graph.split_ids_.clear();
  for (const auto& id : split_ids_) {
    graph.split_ids_.push_back(id);
  }
  graph.exec_count_ = exec_count_;
  return graph;
}

auto Graph::find(std::string name) noexcept -> std::optional<NodeType*> {
  for (auto& node : nodes_) {
    if (std::strcmp(node->info_->name.c_str(), name.c_str()) == 0) {
      return std::make_optional<NodeType*>(node);
    }
  }
  return std::nullopt;
}

auto Graph::find_last_of(std::string name) noexcept
  -> std::optional<NodeType*> {
  for (int i = nodes_.size() - 1; i >= 0; --i) {
    auto* node = nodes_[i];
    if (std::strcmp(node->info_->name.c_str(), name.c_str()) == 0) {
      return std::make_optional<NodeType*>(node);
    }
  }
  return std::nullopt;
}

/*==--- [private] ----------------------------------------------------------==*/

auto Graph::connect(Graph& graph) noexcept -> void {
  const size_t start = std::min(split_ids_.back(), join_ids_.back());
  const size_t end   = std::max(graph.split_ids_[0], graph.join_ids_[0]);

  for (size_t i = 0; i < end; ++i) {
    auto* other_node = graph.nodes_[i];
    for (size_t j = start; j < nodes_.size(); ++j) {
      auto* this_node = nodes_[j];

      // clang-format off
      const bool unmatched_split = 
        other_node->kind() == this_node->kind() &&
        other_node->id()   != this_node->id()   &&
        this_node->kind()  == NodeKind::split;
      // clang-format on

      if (unmatched_split) {
        continue;
      }
      this_node->add_successor(*other_node);
    }
  }
}

auto Graph::find_in_last_split(typename NodeInfo::IdType id) noexcept
  -> std::optional<NodeType*> {
  const int start = split_ids_[split_ids_.size() - 2];
  const int end   = split_ids_[split_ids_.size() - 1];
  for (int i = start; i < end; ++i) {
    auto* node = nodes_[i];
    if (node->id() == id) {
      return std::make_optional<NodeType*>(node);
    }
  }
  return std::nullopt;
}

auto Graph::setup_split_node(NodeType& node) noexcept -> void {
  /* For a split node, we need to find all the indices in the previous split
   * and for any node with the same id, we need to add dependencies between
   * that node and this node. We need to do the same for any friends of the
   * dependents. */
  if (!(node.kind() == NodeKind::split && split_ids_.size() > 1)) {
    return;
  }

  const int start = split_ids_[split_ids_.size() - 2];
  const int end   = split_ids_[split_ids_.size() - 1];
  for (int i = start; i < end; ++i) {
    auto* other = nodes_[i];
    if (
      other->kind() != NodeKind::sync &&
      (other->kind() != NodeKind::split || other->id() != node.id())) {
      continue;
    }
    other->add_successor(node);

    for (const auto& friend_id : other->friends()) {
      if (auto friend_node = find_in_last_split(friend_id)) {
        friend_node.value()->add_successor(node);
      }
    }
  }
}

auto Graph::setup_nonsplit_node(NodeType& node) noexcept -> void {
  if (join_ids_.size() <= 1) {
    return;
  }

  /* This node is a successor of all nodes between the the node with the
   * last index in the join index vector and the last node in the node
   * container, so we need to set the number of dependents for the node and
   * also add this node as a successor to those other nodes, if there are
   * enough join indices. */
  constexpr auto split = NodeKind::split;
  const int      start = join_ids_[join_ids_.size() - 2];
  const int      end   = join_ids_[join_ids_.size() - 1];
  for (int i = start; i < end; ++i) {
    auto& other_node = *nodes_[i];
    if (other_node.kind() == split && node.kind() == split) {
      continue;
    }

    // One of the node kinds is not split, so there is a dependence.
    other_node.add_successor(node);
  }
}

} // namespace ripple
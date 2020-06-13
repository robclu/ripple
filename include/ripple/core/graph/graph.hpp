//==--- ripple/core/graph/graph.hpp ------------------------ -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  graph.hpp
/// \brief This file implements a Graph class, which is simply a collection of
///        nodes with a specifier for where to allocate the nodes.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_GRAPH_GRAPH_HPP
#define RIPPLE_GRAPH_GRAPH_HPP

#include "node.hpp"
#include <ripple/core/algorithm/unrolled_for.hpp>
#include <ripple/core/allocation/allocator.hpp>
#include <ripple/core/arch/cache.hpp>
#include <ripple/core/utility/type_traits.hpp>
#include <memory>
#include <mutex>

namespace ripple {

/// Forward declaration of a graph builder.
class GraphExecutor;

/// The Graph class is simply a collection of nodes, allocated by the allocator.
/// The graph can be executed by an Executor which will schedule the nodes to be
/// run.
class Graph {
  /// Allow the executor access to the graph to schedule correctly.
  friend class GraphExecutor;

  //==--- [constants & aliases] --------------------------------------------==//

  // clang-format off
  /// The alignment for the nodes.
  static constexpr size_t node_alignment = avoid_false_sharing_size;
  /// The default number of nodes.
  static constexpr size_t default_nodes  = 1000;

  /// Defines the type of the nodes for the graph.
  using node_t           = Node<node_alignment>;
  /// The arena for the allocator. We define this here to explicitly use a heap
  /// arena because the graph builder will potentially have to build a large
  /// number of nodes.
  using arena_t          = HeapArena;
  /// Defines the type of the node allocator.
  using node_allocator_t = ThreadSafeObjectPoolAllocator<node_t, arena_t>;
  /// Defines the container user for the nodes in the graph.
  using node_container_t = std::vector<std::unique_ptr<node_t>>;
  /// Defines a container to store the locations of joins.
  using join_container_t = std::vector<int>;
  // clang-format on

  /// Defines a valid type if the type T is not a node type.
  /// \tparam T The type to base the enable on.
  template <typename T>
  using non_node_enable_t =
    std::enable_if_t<!std::is_same_v<node_t, std::decay_t<T>>, int>;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Creates a graph with the given allocator.
  /// \param allocator The allocator for the graph.
  Graph() = default;

  /// Destructor -- recycles the nodes into the pool if they are valid.
  ~Graph() noexcept {
    reset();
  }

  /// Move constructor which just moves the node container from the \p other
  /// graph into this one.
  /// \param other The other graph to move into this one.
  Graph(Graph&& other) noexcept {
    _nodes    = std::move(other._nodes);
    _join_ids = std::move(other._join_ids);
  }

  /// Copy constructor -- deleted because graphs can't be copied. Copying is
  /// expensive, so to create a copy, clone() can be used, which will allocate a
  /// graph of the same form.
  Graph(const Graph&) = delete;

  //==--- [operator overloads] ---------------------------------------------==//

  /// Overload of move assignment operator to move the \p other nodes into this
  /// graph.
  /// \param other The other graph to move into this one.
  auto operator=(Graph&& other) noexcept -> Graph& {
    if (this != &other) {
      _nodes    = std::move(other._nodes);
      _join_ids = std::move(other._join_ids);
    }
    return *this;
  }

  ///  assignment operator to move the \p other nodes into this
  /// graph.
  /// \param other The other graph to move into this one.

  /// Copy assignment operator -- deleted because graphs can't be copied.
  /// Copying is expensive, so to create a copy, clone() can be used, which will
  /// allocate a graph of the same form.
  auto operator=(const Graph& other) = delete;

  //==--- [methods] --------------------------------------------------------==//

  /// Sets the max number of nodes which be allocated quickly for all graphs
  /// combined.
  /// Sets the size of the allocation pool for all grphs combined. It returns
  /// true if the size was set, otherwise returns false. This will only return
  /// true the __first__ time this is called.
  /// \param nodes The number of nodes for the allocation pool.
  static auto set_allocation_pool_size(size_t nodes) -> bool {
    std::lock_guard<std::mutex> g(initialization_mutex());
    if (is_initialized()) {
      return false;
    }

    allocation_pool_nodes() = nodes;
    is_initialized()        = true;
    return true;
  }

  /// Makes a node with the \p callable and \p args.
  /// \param  callable The callable which defines the node's operation.
  /// \param  args     The args for the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The type of the args.
  template <typename F, typename... Args>
  static auto make_node(F&& callable, Args&&... args) -> node_t& {
    return *node_allocator().create<node_t>(
      std::forward<F>(callable), std::forward<Args>(args)...);
  }

  /// Resets the graph, returning all allocated nodes to the allocator.
  auto reset() noexcept -> void {
    for (auto& node : _nodes) {
      if (node) {
        node_allocator().recycle(node.release());
      }
    }
  }

  /// Clones the graph, returning a graph with the same form. This allocates new
  /// nodes for the new graph.
  auto clone() const noexcept -> Graph {
    Graph graph;
    for (const auto& node : _nodes) {
      graph._nodes.emplace_back(node_allocator().create<node_t>(*node));
    }
    graph._join_ids.clear();
    for (const auto& id : _join_ids) {
      graph._join_ids.push_back(id);
    }
    return graph;
  }

  /// Returns the number of nodes in the graph.
  auto size() const -> size_t {
    return _nodes.size();
  }

  /// Returns the maximum number of nodes for __all__ graphs.
  auto allocation_pool_size() const -> size_t {
    return allocation_pool_nodes();
  }

  //==--- [creation interface] ---------------------------------------------==//

  /// Emplaces a node into the graph, returning a reference to the node.
  /// \param  callable The callable which defines the node's operation.
  /// \param  args     Arguments for the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments.
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto emplace(F&& callable, Args&&... args) -> Graph& {
    // Here the allocator doesn't fail, so we don't need make_unique.
    setup_node(*_nodes.emplace_back(node_allocator().create<node_t>(
      std::forward<F>(callable), std::forward<Args>(args)...)));
    return *this;
  }

  /// Emplaces the \p nodes into the graph, returning a reference to each of the
  /// emplaced nodes. This is only enabled if the \p nodes are all Node types,
  /// and should be made using `Graph::make_node()`, for example:
  ///
  /// ~~~{.cpp}
  /// Graph g;
  /// int y = 0;
  /// auto [a, b, c] = graph.emplace_nodes(
  ///   Graph::make_node([] (auto x) { x++; }, y),
  ///   Graph::make_node([] (auto x) { x++; }, y),
  ///   Graph::make_node([] (auto x) { x *= 2; }, y));
  /// ~~~
  ///
  /// The references to the nodes are returned as a tuple.
  ///
  /// \param  nodes The nodes to emplace.
  /// \tparam Nodes The types of the nodes.
  template <
    typename... Nodes,
    all_same_enable_t<node_t, std::decay_t<Nodes>...> = 0>
  auto emplace(Nodes&&... nodes) -> Graph& {
    constexpr size_t node_count = sizeof...(Nodes);

    // Make sure that we have a tuple of __references__:
    auto node_tuple =
      std::tuple<std::decay_t<Nodes>&...>{std::forward<Nodes>(nodes)...};
    unrolled_for<node_count>([&](auto i) {
      setup_node(*_nodes.emplace_back(&std::get<i>(node_tuple)));
    });
    return *this;
  }

  /// Emplaces a node into the graph which runs after all currently emplaced
  /// nodes.
  /// \param  callable The callable which defines the node's operation.
  /// \param  args     Arguments for the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments.
  template <typename F, typename... Args>
  auto then(F&& callable, Args&&... args) -> Graph& {
    _join_ids.emplace_back(_nodes.size());
    // Here the allocator doesn't fail, so we don't need make_unique.
    setup_node(*_nodes.emplace_back(node_allocator().create<node_t>(
      std::forward<F>(callable), std::forward<Args>(args)...)));
    return *this;
  }

 private:
  //==--- [members] --------------------------------------------------------==//

  node_container_t _nodes    = {};     //!< Nodes for the graph.
  join_container_t _join_ids = {1, 0}; //!< Indices of join nodes.

  /// Sets up the node. Initializing the dependency count and successors.
  /// \param node The node to setup for emplacement.
  auto setup_node(node_t& node) -> void {
    // This node is a successor of all nodes between the last index in join
    // index vector, and the last node in the node container, so we need to set
    // the number of dependents for the node, and also add this node as a
    // successor to the other nodes.
    if (_join_ids.size() > 1) {
      const int start = _join_ids[_join_ids.size() - 2];
      const int end   = _join_ids[_join_ids.size() - 1];
      for (int i = start; i < end; ++i) {
        auto& other_node = *_nodes[i];
        other_node.add_successor(node);
        node.increment_num_dependents();

        // If the node doesn't have a parent, then this other node becomes the
        // parent.
        if (node._parent == nullptr) {
          node._parent = &other_node;
        }
      }
    }
  }

  //==--- [static methods] -------------------------------------------------==//

  /// Returns a reference to the allocator for the nodes.
  static auto node_allocator() noexcept -> node_allocator_t& {
    static node_allocator_t allocator(allocation_pool_nodes() * sizeof(node_t));
    return allocator;
  }

  /// Returns a reference to the number of nodes in the allocation pool.
  static auto allocation_pool_nodes() noexcept -> size_t& {
    static size_t nodes_in_pool{default_nodes};
    return nodes_in_pool;
  }

  /// Returns if the builder has been initialized already.
  static auto is_initialized() noexcept -> bool& {
    static bool initialized{false};
    return initialized;
  }

  /// Returns a reference to the initialization mutex.
  static auto initialization_mutex() noexcept -> std::mutex& {
    static std::mutex m;
    return m;
  }
};

} // namespace ripple

#endif // RIPPLE_GRAPH_GRAPH_HPP
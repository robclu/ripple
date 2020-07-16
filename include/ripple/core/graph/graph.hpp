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
#include "splitter.hpp"
#include "reducer.hpp"
#include <ripple/core/algorithm/unrolled_for.hpp>
#include <ripple/core/allocation/allocator.hpp>
#include <ripple/core/arch/cache.hpp>
#include <ripple/core/utility/type_traits.hpp>
#include <memory>
#include <mutex>
#include <optional>

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// Forward declaration of the Graph class.
class Graph;

/// Forward declaration of the execution function for a graph.
static inline auto execute(Graph& g) noexcept -> void;

/// Forward declaration of a graph builder.
class GraphExecutor;

//==--- [implementation] ---------------------------------------------------==//

/// The Graph class is simply a collection of nodes, allocated by the allocator.
/// The graph can be executed by an Executor which will schedule the nodes to be
/// run.
class Graph {
  /// Allow the executor access to the graph to schedule correctly.
  friend class GraphExecutor;
  /// Allow the slitter access to modify the graph.
  friend struct Splitter;

  //==--- [constants & aliases] --------------------------------------------==//

  // clang-format off
  /// The alignment for the nodes.
  static constexpr size_t node_alignment = avoid_false_sharing_size;

  /// The arena for the allocator. We define this here to explicitly use a heap
  /// arena because the graph builder will potentially have to build a large
  /// number of nodes.
  using arena_t               = HeapArena;
  /// Defines the type of the nodes for the graph.
  using node_t                = Node<node_alignment>;
  /// Defines the info type for the nodes.
  using info_t                = NodeInfo;
  /// Defines the type of the node allocator.
  using node_allocator_t      = ThreadSafeObjectPoolAllocator<node_t, arena_t>;
  /// Defines the type of the info allocator.
  using info_allocator_t      = ThreadSafeObjectPoolAllocator<info_t, arena_t>;
  /// Defines the container used for the nodes in the graph.
  using node_container_t      = std::vector<node_t*>;
  /// Defines a container to store the locations of joins.
  using id_container_t        = std::vector<int>;
  // clang-format on

  /// Defines a valid type if the type T is not a node type.
  /// \tparam T The type to base the enable on.
  template <typename T>
  using non_node_enable_t =
    std::enable_if_t<!std::is_same_v<node_t, std::decay_t<T>>, int>;

 public:
  // clang-format off
  /// The default number of nodes.
  static constexpr size_t default_nodes = 1000;
  /// Default id for a node.
  static constexpr auto   default_id    = info_t::default_id;
  // clang-format on

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
        info_allocator().recycle(node->_info);
        node_allocator().recycle(node);
      }
    }
  }

  /// Clones the graph, returning a graph with the same form. This allocates new
  /// nodes for the new graph.
  auto clone() const noexcept -> Graph {
    Graph graph;
    for (const auto& node : _nodes) {
      // TODO: Add copying of node info for new node.
      graph._nodes.emplace_back(node_allocator().create<node_t>(*node));
      graph._nodes.back()->_info =
        info_allocator().create<info_t>(node->_info->name, node->_info->id);
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

  //==--- [find] -----------------------------------------------------------==//

  /// Returns a reference to the node with \p name.
  /// \param name The name of the node to find.
  auto find(std::string name) noexcept -> std::optional<node_t*> {
    for (auto& node : _nodes) {
      if (std::strcmp(node->_info->name.c_str(), name.c_str()) == 0) {
        return std::make_optional<node_t*>(node);
      }
    }
    return std::nullopt;
  }

  /// Returns a reference to the node with \p name.
  /// \param name The name of the node to find.
  auto find_last_of(std::string name) noexcept -> std::optional<node_t*> {
    for (int i = _nodes.size() - 1; i >= 0; --i) {
      auto* node = _nodes[i];
      if (std::strcmp(node->_info->name.c_str(), name.c_str()) == 0) {
        return std::make_optional<node_t*>(node);
      }
    }
    return std::nullopt;
  }

  //==--- [creation interface] ---------------------------------------------==//

  /// Emplaces a node into the graph, returning a reference to the modified
  /// graph.
  /// \param  callable The callable which defines the node's operation.
  /// \param  args     Arguments for the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments.
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto emplace(F&& callable, Args&&... args) -> Graph& {
    const auto no_name = std::string("");
    return emplace_named(
      no_name, std::forward<F>(callable), std::forward<Args>(args)...);
  }

  /// Emplaces a node into the graph with name \p name, returning a reference to
  /// the modified graph.
  /// \param  name     The name of the node to emplace.
  /// \param  callable The callable which defines the node's operation.
  /// \param  args     Arguments for the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments.
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto emplace_named(NodeInfo info, F&& callable, Args&&... args) -> Graph& {
    // Here the allocator doesn't fail, so we don't need make_unique.
    auto& node = *_nodes.emplace_back(node_allocator().create<node_t>(
      std::forward<F>(callable), std::forward<Args>(args)...));
    node._info = info_allocator().create<info_t>(info.name, info.id, info.kind);
    setup_node(node);
    return *this;
  }

  /// Emplaces the \p nodes into the graph, which have no ordering dependency
  /// between them (they can run in parallel).
  ///
  /// ~~~{.cpp}
  /// Graph g;
  /// g.emplace_nodes(
  ///   Graph::make_node([] (auto x) { x++; }, y),
  ///   Graph::make_node([] (auto x) { x++; }, y),
  ///   Graph::make_node([] (auto x) { x *= 2; }, y));
  /// ~~~
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

  /// Emplaces a node into the graph which creates a sync point in the graph.
  ///
  /// \param  callable The callable which defines the node's operation.
  /// \param  args     Arguments for the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments.
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto sync(F&& callable, Args&&... args) -> Graph& {
    _join_ids.emplace_back(_nodes.size());
    NodeInfo info;
    info.kind = NodeKind::sync;
    return emplace_named(
      info, std::forward<F>(callable), std::forward<Args>(args)...);
    return *this;
  }

  /// Emplaces a node into the graph which runs after all currently emplaced
  /// nodes, i.e the operations run synchronously.
  ///
  /// ~~~{.cpp}
  /// Graph g;
  /// g.emplace([] { printf("A\n"); })
  ///  .then([] { printf("B\n"); });
  /// ~~~
  ///
  /// This will run the two operations synchronously.
  ///
  /// \param  callable The callable which defines the node's operation.
  /// \param  args     Arguments for the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments.
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto then(F&& callable, Args&&... args) -> Graph& {
    _join_ids.emplace_back(_nodes.size());
    return emplace(std::forward<F>(callable), std::forward<Args>(args)...);
  }

  /// Emplaces nodes into the graph which run asynchronously with each other,
  /// but sequentially will all previous nodes.
  /// nodes.
  ///
  /// ~~~{.cpp}
  /// Graph g;
  /// g.emplace([] { printf("A\n"); })
  ///  .then(
  ///    Graph::make_node([] { printf("B\n"); }),
  ///    Graph::make_node([] { printf("C\n"); }),
  ///    Graph::make_node([] { printf("D\n"); }));
  /// ~~~
  ///
  /// This will run A, and then B, C, and D will run __after__ A, but
  /// may run in paralell with each other.
  ///
  /// \param  nodes The nodes to emplace.
  /// \tparam Nodes The types of the nodes.
  template <
    typename... Nodes,
    all_same_enable_t<node_t, std::decay_t<Nodes>...> = 0>
  auto then(Nodes&&... nodes) -> Graph& {
    _join_ids.emplace_back(_nodes.size());
    return emplace(std::forward<Nodes>(nodes)...);
  }

  /// If the \p pred returns _true_, then the graph is _scheduled_ to be
  /// executed.
  /// \param  pred  The predicate which returns if the execution must end.
  /// \param  args  The arguments for the predicate.
  /// \tparam Pred  The type of the predicate.
  /// \tparam Args  The type of the predicate arguments.
  template <typename Pred, typename... Args>
  auto conditional(Pred&& pred, Args&&... args) -> Graph& {
    return sync(
      [this](auto&& predicate, auto&&... as) {
        if (predicate(std::forward<Args>(as)...)) {
          execute(*this);
        }
      },
      std::forward<Pred>(pred),
      std::forward<Args>(args)...);
  }

  /// Creates a split in the graph by parallelising the \p callable over the \p
  /// args.
  /// \param  callable The operations for each node in the fork.
  /// \param  args     The arguments to the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments for the callable.
  template <typename F, typename... Args>
  auto split(F&& callable, Args&&... args) -> Graph& {
    Splitter::split(
      *this, std::forward<F>(callable), std::forward<Args>(args)...);
    return *this;
  }

  /// Creates a split in the graph by parallelising the \p callable over the \p
  /// args.
  /// \param  callable The operations for each node in the fork.
  /// \param  args     The arguments to the callable.
  /// \tparam F        The type of the callable.
  /// \tparam Args     The types of the arguments for the callable.
  template <typename F, typename... Args>
  auto then_split(F&& callable, Args&&... args) -> Graph& {
    _join_ids.emplace_back(_nodes.size());
    _split_ids.emplace_back(_nodes.size());
    Splitter::split(
      *this, std::forward<F>(callable), std::forward<Args>(args)...);
    return *this;
  }

  /// Performs a reduction of the \p data, if the container is reducible.
  /// \param  data      The data to reduce, if reducible.
  /// \tparam Container The container for the data.
  template <
    typename Container,
    typename Result,
    typename Pred,
    typename... Args>
  auto
  reduce(Container& data, Result& result, Pred&& pred, Args&&... args) noexcept
    -> Graph& {
    using traits_t  = tensor_traits_t<Container>;
    using value_t   = typename traits_t::value_t;
    using reducer_t = Reducer<value_t, traits_t::dimensions>;

    reducer_t::reduce(
      *this,
      data,
      result,
      std::forward<Pred>(pred),
      std::forward<Args>(args)...);

    // Add a final reduction over the results:
    // auto& results = reducer_t::results_container();
    return *this;
    /*
        return then(
          [&result](auto& res, auto&& p, auto&&... as) {
            result = ::ripple::reduce(res, p, as...);
            printf("Finished: %4.4f\n", result);
          },
          reducer_t::results_container(),
          std::forward<Pred>(pred),
          std::forward<Args>(args)...);
    */
  }

  /// Performs a reduction of the \p data, if the container is reducible.
  /// \param  data      The data to reduce, if reducible.
  /// \tparam Container The container for the data.
  template <
    typename Container,
    typename Result,
    typename Pred,
    typename... Args>
  auto then_reduce(
    Container& data, Result& result, Pred&& pred, Args&&... args) noexcept
    -> Graph& {
    _join_ids.emplace_back(_nodes.size());
    _split_ids.emplace_back(_nodes.size());
    return this->reduce(
      data, result, std::forward<Pred>(pred), std::forward<Args>(args)...);
  }

  /// Returns the number of times the graph has been executed.
  auto num_executions() const -> size_t {
    return _exec_count;
  }

 private:
  //==--- [members] --------------------------------------------------------==//

  node_container_t _nodes      = {};     //!< Nodes for the graph.
  id_container_t   _join_ids   = {1, 0}; //!< Indices of join nodes.
  id_container_t   _split_ids  = {1, 0}; //!< Indices of split nodes.
  size_t           _exec_count = 0; //!< Number of executions for the graph.

  /// Returns a reference to the node with \p id id in the last split.
  /// \param id The id of the node to find. If multiple nodes have the
  auto find_in_last_split(typename info_t::id_t id) noexcept
    -> std::optional<node_t*> {
    const int start = _split_ids[_split_ids.size() - 2];
    const int end   = _split_ids[_split_ids.size() - 1];
    for (int i = start; i < end; ++i) {
      auto* node = _nodes[i];
      if (node->id() == id) {
        return std::make_optional<node_t*>(node);
      }
    }
    return std::nullopt;
  }

  /// Sets up the node. Initializing the dependency count and successors.
  /// \param node The node to setup for emplacement.
  auto setup_node(node_t& node) -> void {
    // For a split node, we need to find all the indices in the previous split
    // and for any node with the same id, we need to add dependencies between
    // this node. We need to do the same for any friends of the dependents:
    if (node.kind() == NodeKind::split) {
      // Only if there has been a split:
      if (_split_ids.size() > 1) {
        const int start = _split_ids[_split_ids.size() - 2];
        const int end   = _split_ids[_split_ids.size() - 1];

        for (int i = start; i < end; ++i) {
          auto* other = _nodes[i];
          if (
            other->kind() != NodeKind::sync &&
            (other->kind() != NodeKind::split || other->id() != node.id())) {
            continue;
          }

          // First add this node as a successor:
          other->add_successor(node);

          // Now go through all of others friends, and add this node as
          // successors of them:
          for (const auto& friend_id : other->friends()) {
            if (auto friend_node = find_in_last_split(friend_id)) {
              friend_node.value()->add_successor(node);
            }
          }
        }
      }
    }

    // This node is a successor of all nodes between the last index in join
    // index vector, and the last node in the node container, so we need to set
    // the number of dependents for the node, and also add this node as a
    // successor to the other nodes.
    if (_join_ids.size() > 1) {
      const int start = _join_ids[_join_ids.size() - 2];
      const int end   = _join_ids[_join_ids.size() - 1];
      for (int i = start; i < end; ++i) {
        setup_nonsplit_node(node, *_nodes[i]);
      }
    }
  }

  /// Sets the \p node from the \p other node.
  /// \param node The node to set up.
  /// \param other_node The other node to use to setup the node.
  auto setup_nonsplit_node(node_t& node, node_t& other_node) -> void {
    constexpr auto split = NodeKind::split;
    // If this is a split node, and the other node is a split node, then there
    // is only a dependence if the nodes have the same id.
    if (
      node.kind() != NodeKind::sync && other_node.kind() == split &&
      node.kind() == split) {
      return;
    }

    // One of the node kinds is not split, so there is a dependence.
    other_node.add_successor(node);
  }

  //==--- [static methods] -------------------------------------------------==//

  /// Returns a reference to the allocator for the nodes.
  static auto node_allocator() noexcept -> node_allocator_t& {
    static node_allocator_t allocator(allocation_pool_nodes() * sizeof(node_t));
    return allocator;
  }

  /// Returns a reference to the allocator for the node information. Here we use
  /// this allocator rather than just a vector or something so that the pointer
  /// in the node to the info is not invalidated when need info is added.
  static auto info_allocator() noexcept -> info_allocator_t& {
    static info_allocator_t allocator(allocation_pool_nodes() * sizeof(info_t));
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
}; // namespace ripple

} // namespace ripple

#endif // RIPPLE_GRAPH_GRAPH_HPP
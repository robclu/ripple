//==--- ripple/core/graph/graph.hpp ------------------------ -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
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

#include "memcopy.hpp"
#include "node.hpp"
#include "splitter.hpp"
#include "reducer.hpp"
#include "../algorithm/unrolled_for.hpp"
#include "../allocation/allocator.hpp"
#include "../arch/cache.hpp"
#include "../utility/spinlock.hpp"
#include "../utility/type_traits.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace ripple {

/*==--- [forward declarations] ---------------------------------------------==*/

/* Forward declaration of the Graph class. */
class Graph;

/* Forward declaration of the executor for the graph. */
class Executor;

/* Forward declaration of the execution function for a graph. */
auto execute(Graph& graph) noexcept -> void;

/**
 * The Graph class is a collection of nodes and the connectivity between them.
 */
class Graph {
  /** Allow the executor access to the graph to schedule correctly. */
  friend class Executor;
  /** Allow the slitter access to modify the graph. */
  friend struct Splitter;
  /** Allows the memcopy functor access  to modify the graph. */
  friend struct Memcopy;

  // clang-format off
  /** The alignment for a node, use some multiple of the false sharing size. */
  static constexpr size_t node_alignment   = avoid_false_sharing_size;
  /** The minimum amount of extra padding for node data. */
  static constexpr size_t node_min_storage = 2 * 72;

  /**
   * The arena for the allocator. We define this here to explicitly use a heap
   * arena because the graph builder will potentially have to build a large
   * number of nodes, and we dont want to run out of stack space.
   */
  using ArenaType     = HeapArena;
  /** Defines the type of the nodes for the graph. */
  using NodeType       = Node<node_alignment, node_min_storage>;
  /** Defines the type of the node allocator. */
  using NodeAllocator = ThreadSafeObjectPoolAllocator<NodeType, ArenaType>;
  /** Defines the type of the node info allocator. */
  using InfoAllocator = ThreadSafeObjectPoolAllocator<NodeInfo, ArenaType>;
  /** Defines the container used for the nodes in the graph. */
  using NodeContainer = std::vector<NodeType*>;
  /** Defines a container to store the locations of joins and splits. */
  using Connections   = std::vector<int>;
  /** Defines the type of the lock used for initialization. */
  using Lock          = Spinlock;
  /** Defines the type of the guard used for initialization. */
  using Guard         = std::lock_guard<Lock>;
  // clang-format on

  /**
   * Defines a valid type if the type T is not a node type.
   * \tparam T The type to base the enable on.
   */
  template <typename T>
  using non_node_enable_t =
    std::enable_if_t<!std::is_same_v<NodeType, std::decay_t<T>>, int>;

 public:
  // clang-format off
  /** The default number of nodes per thread. */
  static constexpr size_t default_nodes = 1024;
  /** Default id for a node. */
  static constexpr auto   default_id    = NodeInfo::default_id;
  // clang-format on

  /*==--- [construction] ---------------------------------------------------==*/

  /** Creates a graph. */
  Graph() = default;

  /**
   * Creates a graph with all nodes having a default execution kind defined by
   * the given execution kind.
   * \param exec_kind The kind of the execution for the graph nodes.
   */
  Graph(ExecutionKind exec_kind) noexcept : execution_{exec_kind} {}

  /**
   * Destructor -- recycles the nodes into the pool if they are valid.
   */
  ~Graph() noexcept {
    reset();
  }

  /**
   * Move constructor which just moves all graph infromation from the other
   * graph into thos one.
   * \param other The other graph to move into this one.
   */
  Graph(Graph&& other) noexcept
  : nodes_{ripple_move(other.nodes_)},
    join_ids_{ripple_move(other.join_ids_)},
    split_ids_{ripple_move(other.split_ids_)},
    exec_count_{other.exec_count_},
    execution_{other.execution_} {
    other.exec_count_ = 0;
  }

  /** Copy constructor -- deleted because graphs can't be copied. */
  Graph(const Graph&) = delete;

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Overload of move assignment operator to move the other graph into this one.
   * \param other The other graph to move into this one.
   * \return A reference to the newly created graph.
   */
  auto operator=(Graph&& other) noexcept -> Graph& {
    if (this != &other) {
      nodes_            = ripple_move(other.nodes_);
      join_ids_         = ripple_move(other.join_ids_);
      split_ids_        = ripple_move(other.split_ids_);
      exec_count_       = other.exec_count_;
      execution_        = other.execution_;
      other.exec_count_ = 0;
    }
    return *this;
  }

  /** Copy assignment operator -- deleted because graphs can't be copied. */
  auto operator=(const Graph& other) = delete;

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Sets the size of the allocation pool for all grphs combined.
   * \param nodes The number of nodes for the allocation pool.
   * \return true when called for the first time, false otherwise.
   */
  static auto set_allocation_pool_size(size_t nodes) noexcept -> bool;

  /**
   * Makes a node with the given callable and args.
   * \param  callable The callable which defines the node's operation.
   * \param  args     The args for the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The type of the args.
   * \return A reference to the new node.
   */
  template <typename F, typename... Args>
  static auto make_node(F&& callable, Args&&... args) -> NodeType& {
    return *node_allocator().create<NodeType>(
      ripple_forward(callable), ripple_forward(args)...);
  }

  /**
   * Makes a node with the given callable and args, with a specific execution
   * kind.
   * \param  callable  The callable which defines the node's operation.
   * \param  exec_kind The kind of the execution for the node.
   * \param  args      The args for the callable.
   * \tparam F         The type of the callable.
   * \tparam Args      The type of the args.
   * \return A reference to the new node.
   */
  template <typename F, typename... Args>
  static auto make_node(F&& callable, ExecutionKind exec_kind, Args&&... args)
    -> NodeType& {
    NodeType& node = *node_allocator().create<NodeType>(
      ripple_forward(callable), ripple_forward(args)...);
    node.info_ = info_allocator().create<NodeInfo>(exec_kind);
    return node;
  }

  /**
   * Resets the graph, returning all allocated nodes to the allocator.
   */
  auto reset() noexcept -> void;

  /**
   * Clones the graph by allocates new nodes for the new graph and then copying
   * the others into it.
   *
   * \note This is not designed to be fast, and it is *not* thread safe.
   *
   * \return The new graph.
   */
  auto clone() const noexcept -> Graph;

  /**
   * Gets the number of nodes in the graph.
   * \return The number of nodes in the graph.
   */
  auto size() const -> size_t {
    return nodes_.size();
  }

  /**
   * Gets the maximum number of nodes which can be allocated for *all* graphs.
   * \return The size of the allocation pool.
   */
  auto allocation_pool_size() const noexcept -> size_t {
    return allocation_pool_nodes();
  }

  /*==--- [find] -----------------------------------------------------------==*/

  /**
   * Gets an optional type which may point to the node with the given name.
   * \param name The name of the node to find.
   * \return A valid pointer wrapped in an optional on success.
   */
  auto find(std::string name) noexcept -> std::optional<NodeType*>;

  /**
   * Finds the last instance of a node with the given name.
   * \param name The name of the node to find.
   * \return A valid optional if the node was found.
   */
  auto find_last_of(std::string name) noexcept -> std::optional<NodeType*>;

  /*==--- [emplace] --------------------------------------------------------==*/

  /**
   * Emplaces a node onto the graph, returning a reference to the modified
   * graph.
   * \param  callable The callable which defines the node's operation.
   * \param  args     Arguments for the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto emplace(F&& callable, Args&&... args) -> Graph& {
    const auto no_name = std::string("");
    return emplace_named(
      no_name, ripple_forward(callable), ripple_forward(args)...);
  }

  /**
   * Emplaces a node onto the graph using the given info, returning a reference
   * to the modified graph.
   *
   * \param  info     The info for the node.
   * \param  callable The callable which defines the node's operation.
   * \param  args     Arguments for the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto emplace_named(NodeInfo info, F&& callable, Args&&... args) -> Graph& {
    auto& node = *nodes_.emplace_back(node_allocator().create<NodeType>(
      ripple_forward(callable), ripple_forward(args)...));
    node.info_ = info_allocator().create<NodeInfo>(
      info.name, info.id, info.kind, info.exec);
    setup_node(node);
    return *this;
  }

  /**
   * Emplaces all the nodes onto the graph at the same level such that they
   * can execute in parallel.
   *
   * \param  nodes The nodes to emplace.
   * \tparam Nodes The types of the nodes.
   * \return A reference to the modified graph.
   */
  template <
    typename... Nodes,
    all_same_enable_t<NodeType, std::decay_t<Nodes>...> = 0>
  auto emplace(Nodes&&... nodes) -> Graph& {
    constexpr size_t node_count = sizeof...(Nodes);

    // Make sure that we have a tuple of __references__:
    auto node_tuple =
      std::tuple<std::decay_t<Nodes>&...>{ripple_forward(nodes)...};
    unrolled_for<node_count>([&](auto i) {
      setup_node(*nodes_.emplace_back(&std::get<i>(node_tuple)));
    });
    return *this;
  }

  /**
   * Emplaces the given graph as a subgraph in this graph, which executes
   * *after* any nodes currently in the graph.
   * \param graph The graph to emplace as a subgraph in this graph.
   * \return A reference to the modified graph.
   */
  auto emplace(Graph& graph) noexcept -> Graph& {
    connect(graph);
    return emplace([&graph] { execute(graph); });
  }

  /*==--- [sync] -----------------------------------------------------------==*/

  /**
   * Emplaces a node into the graph which creates a sync point in the graph.
   *
   * \note This does not flush all gpu work, it simply ensures that all GPU
   *       work is submitted by this point, to flush and wait for all GPU
   *       work to finish, \sa sync_gpus.
   *
   * \param  callable The callable which defines the node's operation.
   * \param  args     Arguments for the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto sync(F&& callable, Args&&... args) -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    NodeInfo info{NodeKind::sync, execution_};
    return emplace_named(
      info, ripple_forward(callable), ripple_forward(args)...);
  }

  /*
   * \note These synchronization functions need to be implemented here, to
   *       ensure that the cuda functions are called when device functionality
   *       is required.
   *
   *       If in a cpp file, then if compiled as c++ code and linked against
   *       a cuda executable, the cuda synchronization won't run.
   */

  /**
   * Emplaces a node into the graph which creates a sync point in the graph
   * which synchronizes *all* streams on *all* GPUs. To create only a submission
   * synchronization point, \sa sync.
   *
   * \param  callable The callable which defines the node's operation.
   * \param  args     Arguments for the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments.
   */
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto sync_gpus(F&& callable, Args&&... args) -> Graph& {
    join_ids_.emplace_back(nodes_.size());

    NodeInfo info{NodeKind::normal, ExecutionKind::gpu};
    for (const auto& gpu : topology().gpus) {
      emplace_named(info, [&gpu] { gpu.synchronize(); });
    }

    join_ids_.emplace_back(nodes_.size());
    info.kind = NodeKind::sync;
    return emplace_named(
      info, ripple_forward(callable), ripple_forward(args)...);
  }

  /**
   * Emplaces a node into the graph which creates a fence for each gpu,
   * which will block until all gpus have finished their currently submitted
   * work.
   *
   * \param  callable The callable which defines the node's operation.
   * \param  args     Arguments for the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments.
   */
  auto gpu_fence() -> Graph& {
    join_ids_.emplace_back(nodes_.size());

    NodeInfo info{NodeKind::normal, ExecutionKind::gpu};
    for (auto& gpu : topology().gpus) {
      gpu.prepare_fence();
      emplace_named(info, [&gpu] { gpu.execute_fence(); });
    }

    join_ids_.emplace_back(nodes_.size());
    info.kind = NodeKind::sync;
    return emplace_named(info, [] {
      size_t count = topology().num_gpus();
      while (count > 0) {
        for (const auto& gpu : topology().gpus) {
          if (gpu.is_fence_down()) {
            count--;
          }
        }
      }
    });
  }

  /*==--- [then] -----------------------------------------------------------==*/

  /**
   * Emplaces a node onto the graph which runs after all currently emplaced
   * nodes, i.e the operations being emplaced run synchronously with the
   * previously emplaced nodes.
   *
   * ~~~{.cpp}
   * Graph g;
   * g.emplace([] { printf("A\n"); })
   *  .then([] { printf("B\n"); });
   * ~~~
   *
   * This will run B *after* A finishes.
   *
   * \param  callable The callable which defines the node's operation.
   * \param  args     Arguments for the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  auto then(F&& callable, Args&&... args) -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    return emplace(ripple_forward(callable), ripple_forward(args)...);
  }

  /**
   * Emplaces nodes onto the graph which run asynchronously with each other,
   * but synchronously with all previous nodes.
   *
   *
   * ~~~{.cpp}
   * Graph g;
   * g.emplace([] { printf("A\n"); })
   *  .then(
   *    Graph::make_node([] { printf("B\n"); }),
   *    Graph::make_node([] { printf("C\n"); }),
   *    Graph::make_node([] { printf("D\n"); }));
   * ~~~
   *
   * This will run A, and then B, C, and D will run *after* A, but may run in
   * parallel with each other.
   *
   * \note This is only enabled when all the template types are the same and
   *       are Node types.
   *
   * \param  nodes The nodes to emplace.
   * \tparam Nodes The types of the nodes.
   * \return A reference to the modified graph.
   */
  template <
    typename... Nodes,
    all_same_enable_t<NodeType, std::decay_t<Nodes>...> = 0>
  auto then(Nodes&&... nodes) -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    return emplace(ripple_forward(nodes)...);
  }

  /**
   * Emplaces the given graph as a subgraph in this graph, which executes after
   * any nodes currently in the graph.
   *
   * \note Any changes to the given graph before this node executes will be
   *       reflected during execution.
   *
   * \param graph The graph to emplace as a subgraph in this graph.
   * \return A reference to the modified graph.
   */
  auto then(Graph& graph) noexcept -> Graph& {
    return then([&graph] { execute(graph); });
  }

  /*==--- [split] ----------------------------------------------------------==*/

  /**
   * Adds a conditional node to the graph, where execution depends on the
   * predicate.
   *
   * \param  pred  The predicate which returns if the execution must end.
   * \param  args  The arguments for the predicate.
   * \tparam Pred  The type of the predicate.
   * \tparam Args  The type of the predicate arguments.
   * \return A reference to the modified graph.
   */
  template <typename Pred, typename... Args>
  auto conditional(Pred&& pred, Args&&... args) -> Graph& {
    return sync(
      [this](auto&& predicate, auto&&... as) {
        if (predicate(ripple_forward(as)...)) {
          execute(*this);
        }
      },
      ripple_forward(pred),
      ripple_forward(args)...);
  }

  /**
   * Creates a split in the graph by parallelising the callable over the args.
   * For any of the args which are tensors, this creates a node per partition
   * in the tensor. Additionally, if any of the modifiers are used, this may
   * place additional nodes into the graph for data transfer of padding data
   * for neighbouring tensor partitions.
   *
   * \note This uses the default execution kind of the graph, so the nodes will
   *       be executed on the CPU or GPU based on the graph execution kind.
   *
   * \param  callable The operations for each node in the split.
   * \param  args     The arguments to the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments for the callable.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args>
  auto split(F&& callable, Args&&... args) noexcept -> Graph& {
    Splitter::split(
      *this, execution_, ripple_forward(callable), ripple_forward(args)...);
    return *this;
  }

  /**
   * Creates a split in the graph by parallelising the callable over the args.
   * For any of the args which are tensors, this creates a node per partition
   * in the tensor. Additionally, if any of the modifiers are used, this may
   * place additional nodes into the graph for data transfer of padding data
   * for neighbouring tensor partitions.
   *
   * \param  exec_kind The kind of execution for the nodes.
   * \param  callable  The operations for each node in the split.
   * \param  args      The arguments to the callable.
   * \tparam F         The type of the callable.
   * \tparam Args      The types of the arguments for the callable.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args>
  auto split(ExecutionKind exec_kind, F&& callable, Args&&... args) noexcept
    -> Graph& {
    Splitter::split(
      *this, exec_kind, ripple_forward(callable), ripple_forward(args)...);
    return *this;
  }

  /**
   * Creates a split in the graph by parallelising the callable over the args.
   * For any of the args which are tensors, this creates a node per partition
   * in the tensor. Additionally, if any of the modifiers are used, this may
   * place additional nodes into the graph for data transfer of padding data
   * for neighbouring tensor partitions.
   *
   * This will add all emplaced nodes such that they are only executed *after*
   * any nodes in the previous levels.
   *
   * \note Nodes assosciated with *the same* partition are the *only* nodes
   *       with dependencies, for example, nodes associated with paritions
   *       0 and 1 will not have any dependencies, and can run in parallel,
   *       *unless* there are padding data transfers required.
   *
   * \note This uses the default execution kind of the graph, so the nodes will
   *       be executed on the CPU or GPU based on the graph execution kind.
   *
   *
   * \param  callable The operations for each node in the split.
   * \param  args     The arguments to the callable.
   * \tparam F        The type of the callable.
   * \tparam Args     The types of the arguments for the callable.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args>
  auto then_split(F&& callable, Args&&... args) noexcept -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    split_ids_.emplace_back(nodes_.size());
    Splitter::split(
      *this, execution_, ripple_forward(callable), ripple_forward(args)...);
    return *this;
  }

  /**
   * Creates a split in the graph by parallelising the callable over the args.
   * For any of the args which are tensors, this creates a node per partition
   * in the tensor. Additionally, if any of the modifiers are used, this may
   * place additional nodes into the graph for data transfer of padding data
   * for neighbouring tensor partitions.
   *
   * This will add all emplaced nodes such that they are only executed *after*
   * any nodes in the previous levels.
   *
   * \note Nodes assosciated with *the same* partition are the *only* nodes
   *       with dependencies, for example, nodes associated with paritions
   *       0 and 1 will not have any dependencies, and can run in parallel,
   *       *unless* there are padding data transfers required.
   *
   *
   * \param  exec_kind The kind of the execution for the operations.
   * \param  callable  The operations for each node in the split.
   * \param  args      The arguments to the callable.
   * \tparam F         The type of the callable.
   * \tparam Args      The types of the arguments for the callable.
   * \return A reference to the modified graph.
   */
  template <typename F, typename... Args>
  auto
  then_split(ExecutionKind exec_kind, F&& callable, Args&&... args) noexcept
    -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    split_ids_.emplace_back(nodes_.size());
    Splitter::split(
      *this, exec_kind, ripple_forward(callable), ripple_forward(args)...);
    return *this;
  }

  /*==--- [memcpy] ---------------------------------------------------------==*/

  /**
   * Performs a copy of the data between the partitions of the data, if the
   * data is a tensor and has partitions, otherwise does not emplace any
   * nodes into the graph.
   *
   * \note If the tensor is wrapped in any modifiers, these are applied as well
   *       to ensure that the dependencies between nodes are created.
   *
   * \note This uses the default execution kind of the graph.
   *
   * \param  args The arguments to the apply the memcopies to.
   * \tparam Args The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename... Args>
  auto memcopy_padding(Args&&... args) noexcept -> Graph& {
    Memcopy::memcopy(
      *this, execution_, TransferKind::asynchronous, ripple_forward(args)...);
    return *this;
  }

  /**
   * Performs a copy of the data between the partitions of the data, if the
   * data is a tensor and has partitions, otherwise does not emplace any
   * nodes into the graph. This operation will only perform the call to start
   * the memcopy operations once all previosuly submitted operations have been
   * executed.
   *
   * \note If the tensor is wrapped in any modifiers, these are applied as well
   *       to ensure that the dependencies between nodes are created.
   *
   * \note This uses the default execution kind of the graph.
   *
   * \param  args The arguments to the apply the memcopies to.
   * \tparam Args The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename... Args>
  auto then_memcopy_padding(Args&&... args) noexcept -> Graph& {
    // join_ids_.emplace_back(nodes_.size());
    // split_ids_.emplace_back(nodes_.size());
    Memcopy::memcopy(
      *this, execution_, TransferKind::synchronous, ripple_forward(args)...);
    return *this;
  }

  /**
   * Performs a copy of the data between the partitions of the data, if the
   * data is a tensor and has partitions, otherwise does not emplace any
   * nodes into the graph.
   *
   * \note If the tensor is wrapped in any modifiers, these are applied as well
   *       to ensure that the dependencies between nodes are created.
   *
   *
   * \param  args The arguments to the apply the memcopies to.
   * \param  exec The kind of the execution of the memory copying, i.e, if
   *              the memcopy is required for the host or device data.
   * \tparam Args The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename... Args>
  auto memcopy_padding(ExecutionKind exec, Args&&... args) noexcept -> Graph& {
    Memcopy::memcopy(*this, exec, ripple_forward(args)...);
    return *this;
  }

  /**
   * Performs a copy of the data between the partitions of the data, if the
   * data is a tensor and has partitions, otherwise does not emplace any
   * nodes into the graph. This operation will only perform the call to start
   * the memcopy operations once all previosuly submitted operations have been
   * executed.
   *
   * \note If the tensor is wrapped in any modifiers, these are applied as well
   *       to ensure that the dependencies between nodes are created.
   *
   *
   * \param  args The arguments to the apply the memcopies to.
   * \param  exec The kind of the execution of the memory copying, i.e, if
   *              the memcopy is required for the host or device data.
   * \tparam Args The types of the arguments.
   * \return A reference to the modified graph.
   */
  template <typename... Args>
  auto
  then_memcopy_padding(ExecutionKind exec, Args&&... args) noexcept -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    split_ids_.emplace_back(nodes_.size());
    Memcopy::memcopy(*this, exec, ripple_forward(args)...);
    return *this;
  }

  /*==--- [reduction] ------------------------------------------------------==*/

  // clang-format off
  /**
   * Performs a reduction of the data using the given predicate.
   * 
   * \note This performs a reduction per partition of the tensor, and then
   *       accumulates the result in the given reduction result.
   * 
   * \note This uses the execution kind of the graph as the execution kind
   *       of the reduction operations.
   * 
   * \param  data   The data to reduce.
   * \param  result The result to place the final value into.
   * \param  pred   The predicate for the reduction.
   * \param  args   Additional arguments for the predicate.
   * \tparam T      The data type for the tensor.
   * \tparam Dims   The number of dimensions for the tensor.
   * \tparam Pred   The type of the predicate.
   * \tparam Args   The type of the arguments for the predicate.
   * \return A reference to the modified graph.
   */
  template <typename T, size_t Dims, typename Pred, typename... Args>
  auto reduce(
    Tensor<T, Dims>&    data,
    ReductionResult<T>& result,
    Pred&&              pred,
    Args&&...           args) noexcept -> Graph& {
    Reducer::reduce(
      *this,
      execution_,
      data,
      result,
      ripple_forward(pred),
      ripple_forward(args)...);

    // Add a synchronization which sets that the reduction is complete.
    return sync([&result] { result.set_finished(); });
  }

  /**
   * Performs a reduction of the data using the given predicate.
   * 
   * \note This performs a reduction per partition of the tensor, and then
   *       accumulates the result in the given reduction result.
   * 
   * \param  exec_kind The kind of the execution of the operations.
   * \param  data      The data to reduce.
   * \param  result    The result to place the final value into.
   * \param  pred      The predicate for the reduction.
   * \param  args      Additional arguments for the predicate.
   * \tparam T         The data type for the tensor.
   * \tparam Dims      The number of dimensions for the tensor.
   * \tparam Pred      The type of the predicate.
   * \tparam Args      The type of the arguments for the predicate.
   * \return A reference to the modified graph.
   */
  template <typename T, size_t Dims, typename Pred, typename... Args>
  auto reduce(
    ExecutionKind       exec_kind,
    Tensor<T, Dims>&    data,
    ReductionResult<T>& result,
    Pred&&              pred,
    Args&&...           args) noexcept -> Graph& {
    Reducer::reduce(
      *this,
      exec_kind,
      data,
      result,
      ripple_forward(pred),
      ripple_forward(args)...);

    // Add a synchronization which sets that the reduction is complete.
    return sync([&result] { result.set_finished(); });
  }

  /**
   * Performs a reduction of the data using the given predicate. This emplces
   * the reduction onto the graph such that it executes *after* any previously
   * emplaced nodes.
   * 
   * \note This performs a reduction per partition of the tensor, and then
   *       accumulates the result in the given reduction result.
   * 
   * \note This uses the execution kind of the graph as the execution kind
   *       of the reduction operations.
   * 
   * \param  data   The data to reduce.
   * \param  result The result to place the final value into.
   * \param  pred   The predicate for the reduction.
   * \param  args   Additional arguments for the predicate.
   * \tparam T      The data type for the tensor.
   * \tparam Dims   The number of dimensions for the tensor.
   * \tparam Pred   The type of the predicate.
   * \tparam Args   The type of the arguments for the predicate.
   * \return A reference to the modified graph.
   */
  template <typename T, size_t Dims, typename Pred, typename... Args>
  auto then_reduce(
    Tensor<T, Dims>&    data,
    ReductionResult<T>& result,
    Pred&&              pred,
    Args&&...           args) noexcept -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    split_ids_.emplace_back(nodes_.size());
    return this->reduce(
      data, result, ripple_forward(pred), ripple_forward(args)...);
  }

  /**
   * Performs a reduction of the data using the given predicate. This emplces
   * the reduction onto the graph such that it executes *after* any previously
   * emplaced nodes.
   * 
   * \note This performs a reduction per partition of the tensor, and then
   *       accumulates the result in the given reduction result.
   * 
   * \note This uses the execution kind of the graph as the execution kind
   *       of the reduction operations.
   * 
   * \param  exec_kind The kind of execution for the operations.
   * \param  data      The data to reduce.
   * \param  result    The result to place the final value into.
   * \param  pred      The predicate for the reduction.
   * \param  args      Additional arguments for the predicate.
   * \tparam T         The data type for the tensor.
   * \tparam Dims      The number of dimensions for the tensor.
   * \tparam Pred      The type of the predicate.
   * \tparam Args      The type of the arguments for the predicate.
   * \return A reference to the modified graph.
   */
  template <typename T, size_t Dims, typename Pred, typename... Args>
  auto then_reduce(
    ExecutionKind       exec_kind,
    Tensor<T, Dims>&    data,
    ReductionResult<T>& result,
    Pred&&              pred,
    Args&&...           args) noexcept -> Graph& {
    join_ids_.emplace_back(nodes_.size());
    split_ids_.emplace_back(nodes_.size());
    return this->reduce(
      exec_kind, data, result, ripple_forward(pred), ripple_forward(args)...);
  }
  // clang-format on

  /**
   * Gets the number of times the graph has been executed.
   * \return The number of times the graph has been executed.
   */
  auto num_executions() const noexcept -> size_t {
    return exec_count_;
  }

 private:
  NodeContainer nodes_      = {};                 //!< Nodes for the graph.
  Connections   join_ids_   = {1, 0};             //!< Indices of join nodes.
  Connections   split_ids_  = {1, 0};             //!< Indices of split nodes.
  size_t        exec_count_ = 0;                  //!< Executions for the graph.
  ExecutionKind execution_  = ExecutionKind::gpu; //!< Default to GPU.

  /**
   * Connects any nodes in the first layer of the given graph to nodes in the
   * last layer of this graph.
   * \param graph The graph to connect to this graph.
   */
  auto connect(Graph& graph) noexcept -> void;

  /**
   * Gets a reference to the node with the given id in the last split.
   * \param id The id of the node to find. If multiple nodes have the
   * \return An optional with a pointer to the node if one was found.
   */
  auto find_in_last_split(typename NodeInfo::IdType id) noexcept
    -> std::optional<NodeType*>;

  /**
   * Sets up the given node. This initializes the dependency count and connects
   * the node to its successors.
   * \param node The node to setup for emplacement.
   */
  auto setup_node(NodeType& node) noexcept -> void {
    setup_split_node(node);
    setup_nonsplit_node(node);
  }

  /**
   * Sets up the node if it is a split node.
   * \param node The node to set up.
   */
  auto setup_split_node(NodeType& node) noexcept -> void;

  /**
   * Node setup for a non-split node.
   * \param node The node to set up.
   */
  auto setup_nonsplit_node(NodeType& node) noexcept -> void;

  /*==--- [static methods] -------------------------------------------------==*/

  /**
   * Gets a reference to the allocator for the nodes.
   * \return A reference to the allocator for the nodes.
   */
  static auto node_allocator() noexcept -> NodeAllocator& {
    static NodeAllocator allocator(allocation_pool_nodes() * sizeof(NodeType));
    return allocator;
  }

  /**
   * Gets a reference to the allocator for the node information. This allocator
   * is used rather than a plain vector or something so that the pointer
   * in the node to the info is not invalidated when need info is added, which
   * is the case for emplacing onto a vector.
   * \return A reference to the info allocator.
   */
  static auto info_allocator() noexcept -> InfoAllocator& {
    static InfoAllocator allocator(allocation_pool_nodes() * sizeof(NodeInfo));
    return allocator;
  }

  /**
   * Gets a reference to the number of nodes in the allocation pool.
   * \return A reference to the number of nodes in the allocation pool.
   */
  static auto allocation_pool_nodes() noexcept -> size_t& {
    static size_t nodes_in_pool{default_nodes};
    return nodes_in_pool;
  }

  /**
   * Determines if the graph has been initialized already.
   * \return true if the graph has been initialized, false otherwise.
   */
  static auto is_initialized() noexcept -> bool& {
    static bool initialized{false};
    return initialized;
  }

  /**
   * Gets a reference to the initialization lock.
   * \return A reference to the initialization lock.
   */
  static auto initialization_lock() noexcept -> Lock& {
    static Lock lock;
    return lock;
  }
};

} // namespace ripple

#endif // RIPPLE_GRAPH_GRAPH_HPP
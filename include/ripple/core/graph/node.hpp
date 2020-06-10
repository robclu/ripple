//==--- ripple/core/graph/node.hpp ------------------------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  node.hpp
/// \brief This file implements a Node class for a graph.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_GRAPH_NODE_HPP
#define RIPPLE_GRAPH_NODE_HPP

#include <ripple/core/container/tuple.hpp>
#include <ripple/core/functional/invocable.hpp>

namespace ripple {

//==--- [node executor] ----------------------------------------------------==//

/// The NodeExecutor struct is a base class which enables Node types to be
/// stored in a single contained when the have different signatures.
struct NodeExecutor {
  // clang-format off
  /// Defaulted constructor.
  NodeExecutor() noexcept          = default;
  /// Virtual destructor to avoid incorrect deletion of base class.
  virtual ~NodeExecutor() noexcept = default;

  /// Copy constructor -- deleted.
  NodeExecutor(const NodeExecutor&) = delete;
  /// Move constructor -- deleted.
  NodeExecutor(NodeExecutor&&)      = delete;

  /// Copy assignment -- deleted.
  auto operator=(const NodeExecutor&) = delete;
  /// Move assignment -- deleted.
  auto operator=(NodeExecutor&&)      = delete;
  // clang-format on

  /// The clone method enables copying/moving the executor.
  /// \param storage A pointer to the storage to clone the executor into.
  virtual auto clone(void* storage) const noexcept -> NodeExecutor* = 0;

  /// The execute method defines the interface for node execution.
  virtual auto execute() noexcept -> void = 0;

  /// The execute method defines the interface for node execution.
  virtual auto execute() const noexcept -> void = 0;
};

//==--- [node executor impl] -----------------------------------------------==//

/// The NodeExecutorImpl struct implements the NodeExecutor interface, but
/// stores the callable object which should be invoked for the node.
/// \tparam   Callable  The type of the invocable to store.
/// \tparam   Args      The type of the callable's arguments to store.
template <typename Callable, typename... Args>
struct NodeExecutorImpl final : public NodeExecutor {
 private:
  // clang-format off
  /// Defines an alias for the type of the executor.
  using invocable_t = Invocable<std::decay_t<Callable>>;
  /// Defines the type of the arguments to pass to the invocable.
  using args_t      = Tuple<Args...>;

  /// The number of arguments for the execution.
  static constexpr size_t num_args = sizeof...(Args);
  // clang-format on

 public:
  // Constructor to store an \p invocable and \p args.
  /// \param  invocable     The invocable object to store.
  /// \param  args          The arguments for the invocable.
  /// \tparam InvocableType The type of the invocable.
  /// \tparam ArgTypes      The types of the arguments.
  template <typename InvocableType, typename... ArgTypes>
  NodeExecutorImpl(InvocableType&& invocable, ArgTypes&&... args) noexcept
  : _invocable{std::forward<InvocableType>(invocable)},
    _args{std::forward<ArgTypes>(args)...} {}

  /// Destuctor -- defaulted.
  ~NodeExecutorImpl() noexcept final = default;

  /// Copy constructor -- copies \p other into this.
  /// \param other The other executor to copy.
  NodeExecutorImpl(const NodeExecutorImpl& other) noexcept
  : _invocable{other._invocable}, _args{other._args} {}

  /// Move constructor -- moves \p other into this.
  /// \param other The other executor to copy.
  NodeExecutorImpl(NodeExecutorImpl&& other) noexcept
  : _invocable{std::move(other._invocable)}, _args{std::move(other._args)} {}

  /// Copy assignment to copy the \p other executor to this one.
  /// \param other The other executor to copy to this one.
  auto operator=(const NodeExecutorImpl& other) noexcept -> NodeExecutorImpl& {
    _invocable = other._invocable;
    _args      = other._args;
    return *this;
  }

  /// Copy assignment to copy the \p other executor to this one.
  /// \param other The other executor to copy to this one.
  auto operator=(NodeExecutorImpl&& other) noexcept -> NodeExecutorImpl& {
    _invocable = std::move(other._invocable);
    _args      = std::move(other._args);
    return *this;
  }

  //==--- [intefacce impl --------------------------------------------------==//

  /// Override of the execute method to run the executor.
  auto execute() noexcept -> void final {
    execute_impl(std::make_index_sequence<num_args>());
  }

  /// Override of the execute method to run the executor.
  auto execute() const noexcept -> void final {
    execute_impl(std::make_index_sequence<num_args>());
  }

  /// Override of the clone method to copy this class into the provided
  /// \p storage.
  /// \param storage  The storage to clone into.
  auto clone(void* storage) const noexcept -> NodeExecutorImpl* final {
    new (storage) NodeExecutorImpl(*this);
    return reinterpret_cast<NodeExecutorImpl*>(storage);
  }

 private:
  invocable_t _invocable; //!< The object to be invoked.
  args_t      _args;      //!< Args for the invocable.

  /// Executes the invocable, expanding the args.
  /// \tparam I The indices for the arguments.
  template <size_t... I>
  auto execute_impl(std::index_sequence<I...>) noexcept -> void {
    _invocable(get<I>(_args)...);
  }

  /// Executes the invocable, expanding the args.
  /// \tparam I The indices for the arguments.
  template <size_t... I>
  auto execute_impl(std::index_sequence<I...>) const noexcept -> void {
    _invocable(get<I>(_args)...);
  }
};

//==--- [node] -------------------------------------------------------------==//

/// Default number of successors for a node.
static constexpr size_t default_successors = 4;

/// A Node is an operation in a graph, which performs some work.
///
/// It should be given an alignment which is a multiple of the cache line size
/// so that there is no false sharing of nodes when they are used across
/// threads.
///
/// There is some overhead for the node which seems unavoidable. In order for
/// the Node's work to be __any__ callable object, the abstract base
/// class NodeExecutor is required, and therefore the task must store a pointer
/// to the executor for the node.
///
/// This is not really a problem in terms of storage overhead since tasks
/// should be cache line aligned, and the 8 bytes are usually only a fraction
/// of the cache line size, but it does reduce the available storage for
/// the callable for the Node's work and the arguments for the callable. It is
/// therefore only a problem if the callable has many arguments, or the
/// arguments are large. If this is a problem, a compile time error will be
/// generated, in which case the alignment of the Node can be increased by the
/// cachline size.
///
/// Executing the Node's work then requires invoking the stored callable
/// through the base NodeExecutor class. Tests have shown that the compiler is
/// usually able to remove the virtual function indirection, and the performance
/// is usually the same as a __non inlined__ function call. The major drawback
///  therefore is the loss of the ability for the compiler to inline.
///
/// However, benchmarking has shown that the the cost is approximately 1.1ns
/// for any body of work executed through the NodeExecutor vs 0.25ns for an
/// inlined version. This cost is acceptable for the felxibility of
/// the genericy of the node interface.
///
/// The other alternative would be to use a variant of different function
/// signatures, however, that also has a runtime cost, so this solution is
/// alright.
///
/// The above limitations are also not significant, since the __correct__ use of
/// nodes in the graph is to perform __non trivial__ workloads. Even a node with
/// a ~30ns workload only incurrs a ~2.5% overhead compared to if the Node's
/// work body was executed inline. Most work should be in the us to ms range
/// anyway, making the overhead negligible.
///
/// \tparam Alignment The aligment for the node.
/// \tparam Successor The number of successors.
template <size_t Alignment, size_t Successors = default_successors>
struct alignas(Alignment) Node {
 public:
  // clang-format off
  /// Defines the type of the unfinished counter.
  using counter_t    = std::atomic<uint16_t>;
  /// Defines the type of the successor array.
  using successors_t = std::array<Node*, Successors>;

 private:
  /// Memory required for the parent and the successors.
  static constexpr size_t node_mem_size = sizeof(Node*) + sizeof(successors_t);
  /// Memory required for all node data.
  static constexpr size_t data_mem_size =
    sizeof(NodeExecutor*) + sizeof(counter_t) + node_mem_size;
  /// Spare space on cache line.
  static constexpr size_t spare_mem     = data_mem_size % Alignment;
  /// Threshold for minimum spare space.
  static constexpr size_t min_spare_mem = 48;
  // clang-format on

  /// Returns true if T is not a Node.
  template <typename T>
  static constexpr bool is_not_node_v = !std::is_same_v<std::decay_t<T>, Node>;

  /// Defines a valid type if T is not a node.
  template <typename T>
  using non_node_enable_t = std::enable_if_t<is_not_node_v<T>, int>;

  //==--- [constants] ------------------------------------------------------==//

 public:
  /// Defines the size of the buffer for the storage.
  static constexpr size_t size =
    spare_mem + (spare_mem >= min_spare_mem ? size_t{0} : Alignment);

  //==--- [members] --------------------------------------------------------==//

  NodeExecutor* executor      = nullptr; //!< The executor to run the task.
  Node*         parent        = nullptr; //!< Id of the task's parent.
  successors_t  successors    = {};      //!< Array of successors.
  counter_t     dependants    = 0;       //!< Counter for dependant
  char          storage[size] = {};      //!< Additional storage.

  //==--- Con/destruction --------------------------------------------------==//

  // clang-format off
  /// Default constructor -- uses the default values of the Node.
  Node() noexcept  = default;
  /// Default destructor.
  ~Node() noexcept = default;
  // clang-format on

  /// Copy constructor which clones the executor and copies the rest of the node
  /// state.
  /// This will check against the executor being a nullptr in debug mode.
  /// \param other The other task to copy from.
  Node(const Node& other) noexcept {
    debug_assert_node_valid(other);
    executor = other.executor->clone(&storage);
    parent   = other.parent;
    dependants.store(
      other.dependants.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    for (int i = 0; i < other.successors.size(); ++i) {
      successors[i] = other.successors[i];
    }
  }

  /// Move constructor which just calls the copy constructor, and invalidates
  /// the \p other node.
  /// \param other The other task to move from.
  Node(Node&& other) noexcept : Node(other) {
    other.executor = nullptr;
    other.parent   = nullptr;
    other.dependants.store(0, std::memory_order_relaxed);
    for (auto& node : other.successors) {
      node = nullptr;
    }
  }

  /// Constructs the executor by storing the \p callable and the callable's
  /// arguments in the additional storage for the node.
  ///
  /// The constructor is only enabled if F is not a Node.
  ///
  /// If the \p callable and its arguments wont fit into the node storage then
  /// a compile time error is generated. The size of the node needs to be
  /// increased by a cacheline size.
  ///
  /// \param  callable The callable object to store.
  /// \param  args     The arguments to store.
  /// \tparam F        The type of the callable object.
  /// \tparam Args     The type of the arguments for the callable.
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  Node(F&& callable, Args&&... args) noexcept {
    set_executor(std::forward<F>(callable), std::forward<Args>(args)...);
  }

  //==--- [operator overloads] ---------------------------------------------==//

  /// Copy assignment overload which clones the executor and copies the node
  /// state.
  /// This will check against the \p other node being a nullptr in debug mode.
  /// \param other The other node to copy.
  auto operator=(const Node& other) noexcept -> Node& {
    if (this == &other) {
      return *this;
    }

    debug_assert_valid_node(other);
    executor = other.executor->clone(&storage);
    parent   = other.parent;
    dependants.store(
      other.dependants.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    for (int i = 0; i < other.successors.size(); ++i) {
      successors[i] = other.successors[i];
    }
    return *this;
  }

  /// Move assignment overload which clones the executor, copies the other node
  /// state, and invalidates the \p other node.
  /// This will check against the \p other node being a nullptr in debug mode.
  /// \param other  The other node to move from.
  auto operator=(Node&& other) noexcept -> Node& {
    if (this == &other) {
      return *this;
    }

    debug_assert_node_valid(other);
    *this          = other;
    other.executor = nullptr;
    other.parent   = nullptr;
    other.dependants.store(0, std::memory_order_relaxed);

    for (auto& node : other.successors) {
      node = nullptr;
    }
    return *this;
  }

  //==--- [methods] --------------------------------------------------------==//

  /// Sets the executor of the node to the callable \p f.
  /// \param  callable The callable object to store.
  /// \param  args     The arguments to store.
  /// \tparam F        The type of the callable object.
  /// \tparam Args     The type of the arguments for the callable.
  template <typename F, typename... Args>
  auto set_executor(F&& callable, Args&&... args) noexcept -> void {
    using executor_t = NodeExecutorImpl<F, Args...>;
    static_assert(
      size >= sizeof(executor_t),
      "Node storage is too small to allocate callable and it's args!");

    new (&storage)
      executor_t(std::forward<F>(callable), std::forward<Args>(args)...);
    executor = reinterpret_cast<executor_t*>(&storage);
  }

 private:
  /// Checks if the \p node is valid. In release mode this is disabled. In
  /// debug mode this will terminate if the executor in \p node is a nullptr.
  /// \param node  The node to check the validity of.
  auto debug_assert_node_valid(const Node& node) const noexcept -> void{};
};

} // namespace ripple

#endif // RIPPLE_GRAPH_NODE_HPP
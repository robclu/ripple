//==--- ripple/core/graph/node.hpp ------------------------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
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
#include <ripple/core/math/math.hpp>
#include <ripple/core/utility/forward.hpp>
#include <array>
#include <atomic>
#include <cassert>
#include <string>
#include <vector>

namespace ripple {

/** Forward declaration of the graph class. */
class Graph;

/** Defines the kinds of nodes. */
enum class NodeKind : uint8_t {
  normal = 0, //!< Default kind of node.
  split  = 1, //!< Node in a split operation in a graph.
  sync   = 2  //!< Node kind to explicity create a sync point in a graph.
};

/*==--- [node executor] ----------------------------------------------------==*/

/**
 * The NodeExecutor struct is a base class which enables a types which derive
 * from it to be stored in a container for execution. Thiis interface defines
 * an execute method which performs execution of the node work.
 */
struct NodeExecutor {
  // clang-format off
  /** Defaulted constructor. */
  NodeExecutor() noexcept          = default;
  /** Virtual destructor to avoid incorrect deletion from derived class. */
  virtual ~NodeExecutor() noexcept = default;

  /** Copy constructor -- deleted. */
  NodeExecutor(const NodeExecutor&) = delete;
  /** Move constructor -- deleted. */
  NodeExecutor(NodeExecutor&&)      = delete;

  /** Copy assignment -- deleted. */
  auto operator=(const NodeExecutor&) = delete;
  /** Move assignment -- deleted. */
  auto operator=(NodeExecutor&&)      = delete;
  // clang-format on

  /**
   * The clone method enables copying/moving the executor, but makes the
   * intention to do so explicit.
   * \param storage A pointer to the storage to clone the executor into.
   * \return A pointer to the cloned executor.
   */
  virtual auto clone(void* storage) const noexcept -> NodeExecutor* = 0;

  /**
   * Executes the work assosciated with the executor.
   */
  virtual auto execute() noexcept -> void = 0;
};

/*==--- [node executor impl] -----------------------------------------------==*/

/**
 * The NodeExecutorImpl struct implements the NodeExecutor interface, storing
 * a callable object which defines the work to be executed.
 *
 * \tparam Callable The type of the invocable to store.
 * \tparam Args     The type of the callable's arguments to store.
 */
template <typename Callable, typename... Args>
struct NodeExecutorImpl final : public NodeExecutor {
 private:
  // clang-format off
  /** Defines an alias for the type of the invocable for the node. */
  using Invocable    = Invocable<std::decay_t<Callable>>;
  /** Defines the type of the argument contianer. */
  using ArgContainer = Tuple<Args...>;

  /** The number of arguments for the execution. */
  static constexpr size_t num_args = sizeof...(Args);
  // clang-format on

 public:
  /**
   * Constructor to store the invocable and args for the executor.
   * \param  invocable     The invocable object to store.
   * \param  args          The arguments for the invocable.
   * \tparam InvocableType The type of the invocable.
   * \tparam ArgTypes      The types of the arguments.
   */
  template <typename InvocableType, typename... ArgTypes>
  NodeExecutorImpl(InvocableType&& invocable, ArgTypes&&... args) noexcept
  : invocable_{ripple_forward(invocable)}, args_{ripple_forward(args)...} {}

  /** Destuctor -- defaulted. */
  ~NodeExecutorImpl() noexcept final = default;

  /**
   * Copy constructor which copies the other executor into this one.
   * \param other The other executor to copy.
   */
  NodeExecutorImpl(const NodeExecutorImpl& other) noexcept
  : invocable_{other.invocable_}, args_{other.args_} {}

  /**
   * Move constructor which moves the other executor into this.
   * \param other The other executor to copy.
   */
  NodeExecutorImpl(NodeExecutorImpl&& other) noexcept
  : invocable_{ripple_move(other._invocable)},
    args_{ripple_move(other._args)} {}

  /**
   * Copy assignment to copy the other executor to this one.
   * \param other The other executor to copy to this one.
   * \return A reference to the new executor.
   */
  auto operator=(const NodeExecutorImpl& other) noexcept -> NodeExecutorImpl& {
    invocable_ = other.invocable_;
    args_      = other.args_;
    return *this;
  }

  /**
   * Move assignment to move the  other executor to this one.
   * \param other The other executor to move to this one.
   * \return A reference to the new executor.
   */
  auto operator=(NodeExecutorImpl&& other) noexcept -> NodeExecutorImpl& {
    if (&other != this) {
      invocable_ = ripple_move(other.invocable_);
      args_      = ripple_move(other.args_);
    }
    return *this;
  }

  /*==--- [interface impl] -------------------------------------------------==*/

  /**
   * Implementation of the execute method to run the executor.
   */
  auto execute() noexcept -> void final {
    execute_impl(std::make_index_sequence<num_args>());
  }

  /**
   * Implementation of the clone method to copy this class into the provided
   * storage.
   * \param storage  The storage to clone into.
   * \return A pointer to the cloned executor.
   */
  auto clone(void* storage) const noexcept -> NodeExecutorImpl* final {
    new (storage) NodeExecutorImpl(*this);
    return reinterpret_cast<NodeExecutorImpl*>(storage);
  }

 private:
  Invocable    invocable_; //!< The object to be invoked.
  ArgContainer args_;      //!< Args for the invocable.

  /**
   * Implementation of execution of the invocable, expanding the args into it.
   * \tparam I The indices for the arguments.
   */
  template <size_t... I>
  auto execute_impl(std::index_sequence<I...>) noexcept -> void {
    invocable_(get<I>(args_)...);
  }
};

/*==--- [node info] --------------------------------------------------------==*/

/**
 * This class defines extra infromation for a node which may be useful for
 * the construction of graphs, but which is not necessary for execution.
 * We store it separately to reduce the memory footprint of the node so that
 * on the fast path (node execution) this information does not result in a
 * performance hit.
 */
struct NodeInfo {
  // clang-format off
  /** Defines the type used for the node name. */
  using Name    = std::string;
  /// Defines the type used for the node id.
  using IdType  = uint64_t;
  /// Defines the type of the friends container.
  using Friends = std::vector<IdType>;

  /// Default id of the node.
  static constexpr IdType default_id   = std::numeric_limits<IdType>::max();
  /// Default name of the node.
  static constexpr auto   default_name = "";
  // clang-format on

  /**
   * Creates an id for a node from the given indices, using hash combining.
   * \param  indices The indices to create an id from.
   * \tparam Size    The number of ids.
   * \return The id of the node.
   */
  template <size_t Size>
  static auto
  id_from_indices(const std::array<uint32_t, Size>& indices) -> uint64_t {
    using namespace math;
    static_assert(Size <= 3, "Node id only valid for up to 3 dimensions!");
    return Size == 1
             ? indices[0]
             : Size == 2
                 ? hash_combine(indices[0], indices[1])
                 : Size == 3
                     ? hash_combine(
                         indices[2], hash_combine(indices[0], indices[1]))
                     : 0;
  }

  /**
   * Creates a name for a node from the indices.
   * \param  indices The indices to create a name from.
   * \tparam Size    The number of ids.
   * \return The name of the node.
   */
  template <size_t Size>
  static auto
  name_from_indices(const std::array<uint32_t, Size>& indices) noexcept
    -> Name {
    Name name = Size == 0 ? "" : std::to_string(indices[0]);
    for (auto i : range(Size - 1)) {
      name += "_" + std::to_string(indices[i + 1]);
    }
    return name;
  }

  /*==--- [construction] ---------------------------------------------------==*/

  /** Default constructor for node info. */
  NodeInfo() = default;

  /**
   * Constructor to set the execution kind for the node.
   * \param exec_kind_ The execution kind for the node.
   */
  explicit NodeInfo(ExecutionKind exec_kind_) noexcept : exec{exec_kind_} {}

  /**
   * Constructor to set the kind and execution kind for the node.
   * \param kind_      The kind of teh node.
   * \param exec_kind_ The execution kind for the node.
   */
  explicit NodeInfo(NodeKind kind_, ExecutionKind exec_kind_) noexcept
  : kind{kind_}, exec{exec_kind_} {}

  /**
   * Constructor for node info which sets the name for the node.
   * \param name_ The name of the node.
   */
  NodeInfo(Name name_) noexcept : name{ripple_move(name_)} {}

  /**
   * Constructor to set the node name and id.
   * \param name_ The name of the node.
   * \param id_   The id of the node.
   */
  NodeInfo(Name name_, IdType id_) noexcept
  : name{ripple_move(name_)}, id{id_} {}

  /**
   * Constructor to set the node name , id, and kind.
   * \param name_ The name of the node.
   * \param id_   The id of the node.
   * \param kind_ The kind of the node.
   */
  NodeInfo(Name name_, IdType id_, NodeKind kind_) noexcept
  : name{ripple_move(name_)}, id{id_}, kind{kind_} {}

  /**
   * Constructor to set the node name, id, kind, and execution target for the
   * node.
   * \param name_ The name of the node.
   * \param id_   The id of the node.
   * \param kind_ The kind of the node.
   * \param exec_ The execution kind of the node.
   */
  NodeInfo(Name name_, IdType id_, NodeKind kind_, ExecutionKind exec_) noexcept
  : name{ripple_move(name_)}, id{id_}, kind{kind_}, exec{exec_} {}

  /*==--- [deleted] --------------------------------------------------------==*/

  // clang-format off
  /** Copy constructor -- deleted. */
  NodeInfo(const NodeInfo& other) = default; 
  /** Move constructor deleted. */
  NodeInfo(NodeInfo&& other)      = default;
  /** Move assignment -- deleted. */
  auto operator=(const NodeInfo&) = delete;
  /** Copy assignment -- deleted. */
  auto operator=(NodeInfo&&)      = delete;
  // clang-format on

  /*==--- [members] --------------------------------------------------------==*/

  Name          name    = default_name;       //!< The name of the node.
  Friends       friends = {};                 //!< Siblings for the node.
  IdType        id      = default_id;         //!< Id of the node.
  NodeKind      kind    = NodeKind::normal;   //!< The kind of the node.
  ExecutionKind exec    = ExecutionKind::gpu; //!< Execution kind of the node.
};

/*==--- [node impl] --------------------------------------------------------==*/

/**
 * Implementation type of a node class, which defines an operation a graph
 * which performs work.
 *
 * It should be given an alignment which is a multiple of the cache line size
 * so that there is no false sharing of nodes when they are used across
 * threads.
 *
 * There is a small amount of overhead in the nodes, but they are still cheap
 * In order for the Node's work to be __any__ callable object, the abstract base
 * class NodeExecutor pointer needs to be stores.
 *
 * This is not really a problem in terms of storage overhead since ndoes are
 * cache line aligned, and the 8 bytes are usually only a fraction of the cache
 * line size, but it does reduce the available storage for the callable for the
 * executor's work and more importantly, the arguments for the callable. It is
 * therefore only a problem if the callable has many arguments, or the
 * arguments are large. If this is a problem, a compile time error will be
 * generated, in which case the alignment of the Node can be increased by the
 * cachline size.
 *
 * Executing the Node's work then requires invoking the stored callable
 * through the base NodeExecutor class. Benchmarks have shown that the compiler
 * is usually able to remove the virtual function indirection, and the
 * performance is usually the same as a __non inlined__ function call. The
 * major drawback therefore is the loss of the ability for the compiler to
 * inline.
 *
 * However, the benchmarking also showed that the the cost is approximately
 * 1.1ns for any body of work executed through the NodeExecutor vs 0.25ns for
 * an inlined version. This cost is therfore pretty small, and worth the added
 * flexibility of allowing nodes to have any signature.
 *
 * \note The benchmarks were also on *very* cheap nodes (with small workloads).
 *
 * The above limitations are also not significant, since the *correct* use of
 * nodes in the graph is to perform *non trivial* workloads. Even a node with
 * a ~30ns workload only incurrs a ~2.5% overhead compared to if the Node's
 * work body was executed inline. Most work should be in the us to ms range
 * anyway, making the overhead negligible.
 *
 * \tparam Alignment  The aligment for the node.
 * \tparam MinStorage The number of bytes of storage required for a node.
 */
template <size_t Alignment, size_t MinStorage = 72>
class alignas(Alignment) Node {
  /** Allow the graph class access to the node for building the graph. */
  friend Graph;

  // clang-format off
  /** 
   * We use a vector for the successors, because it's only 24 bytes and enfore
   * some limits on the number of successors for a node. What we really need to
   * ensure is that the successors are contiguous so that when they are modified
   * or accessed by a node they are on the same cache line, which is what the
   * vector gives us.
   */
  using Successors      = std::vector<Node*>;
  /** Defines the value type of the counter. */
  using CounterValue    = uint32_t;
  /**  Defines the type of the unfinished counter. */
  using Counter         = std::atomic<CounterValue>;
  /** Defines the type of the pointer to the node information. */
  using NodeInfoPtr     = NodeInfo*;
  /** Defines the type of the node executor pointer. */
  using NodeExecutorPtr = NodeExecutor*;

  // clang-format off
  /** Memory required for all node data.  */
  static constexpr size_t data_mem_size =
    sizeof(Successors)      +
    sizeof(NodeExecutorPtr) + 
    sizeof(NodeInfoPtr)     +
    sizeof(CounterValue)    +
    sizeof(Counter);

  /** Spare space on cache line. */
  static constexpr size_t spare_mem  = data_mem_size % Alignment;
  /** Multiples of alignment required for the node. */
  static constexpr size_t align_mult = (data_mem_size + MinStorage) / Alignment;
  /** Defines the size of the storage buffer for the node. */
  static constexpr size_t size       = spare_mem + (align_mult * Alignment);
  // clang-format on

  /** Returns true if T is not a Node. */
  template <typename T>
  static constexpr bool is_not_node_v = !std::is_same_v<std::decay_t<T>, Node>;

  /** Defines a valid type if T is not a node. */
  template <typename T>
  using non_node_enable_t = std::enable_if_t<is_not_node_v<T>, int>;

 public:
  /*==--- [construction] ---------------------------------------------------==*/

  // clang-format off
  /** Default constructor. */
  Node() noexcept  = default;
  /** Default destructor. */
  ~Node() noexcept = default;
  // clang-format on

  /**
   * Copy constructor which clones the executor and copies the rest of the node
   * state.
   *
   * \note This will check against the executor being a nullptr in debug mode,
   *       in release there is not check and a null executor will likely cause
   *       a segfault.
   *
   * \note This could be expensive if the node has *a lot* of successors.
   *
   * \note This does not copy the node's information. Information for a node
   *       is unique and should be allocated and set after copying a noce.
   *
   * \param other The other node to copy into this node.
   */
  Node(const Node& other) noexcept {
    debug_assert_node_valid(other);
    executor_ = other.executor_->clone(&storage_);
    incoming_ = other.incoming_;
    dependents_.store(
      other.dependents_.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    for (size_t i = 0; i < other.successors_.size(); ++i) {
      successors_[i] = other.successors_[i];
    }
  }

  /**
   * Move constructor which just calls the copy constructor for most of the node
   * data, but moves the successors of the other node into this node.
   * \param other The other task to move from.
   */
  Node(Node&& other) noexcept
  : executor_(other.executor_),
    info_(other.info_),
    incoming_(other.incoming_),
    successors_(ripple_move(other.successors_)),
    dependents_(other.dependents_.load(std::memory_order_relaxed)) {
    other._executor = nullptr;
    other._info     = nullptr;
    other._incoming = 0;
    other._dependents.store(0, std::memory_order_relaxed);
  }

  /**
   *  Constructs the node, creating its executor by storing the callable and
   * the callable's arguments in the additional storage for the node.
   *
   * \note This constructor is only enabled if F is not a Node.
   *
   * \note If the callable and its arguments will not fit into the node storage
   *       then this will fail at compile time and the size of the node will
   *       need to be increased.
   *
   *
   * \param  callable The callable object to store.
   * \param  args     The arguments to store.
   * \tparam F        The type of the callable object.
   * \tparam Args     The type of the arguments for the callable.
   */
  template <typename F, typename... Args, non_node_enable_t<F> = 0>
  Node(F&& callable, Args&&... args) noexcept {
    set_executor(ripple_forward(callable), ripple_forward(args)...);
  }

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Copy assignment overload which clones the executor and copies the node
   * state.
   *
   * \note This will check against the other node's executor being a nullptr in
   *       debug, but will likely cause a segfault in release.
   *
   * \note This does not copy the node information, which should be unique
   *       and therefore allocated and then set after copying the node.
   *
   * \param other The other node to copy.
   * \return A reference to the new node.
   */
  auto operator=(const Node& other) noexcept -> Node& {
    if (this == &other) {
      return *this;
    }

    debug_assert_valid_node(other);
    executor_ = other.executor_->clone(&storage_);
    incoming_ = other.incoming_;
    dependents_.store(
      other.dependents_.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    for (int i = 0; i < other.successors_.size(); ++i) {
      successors_[i] = other.successors_[i];
    }
    return *this;
  }

  /**
   * Move assignment overload which clones the executor, copies the other node
   * state, moves the other node's successors, and invalidates the other node.
   *
   * \note This will check against the other node's executor being a nullptr in
   *       debug, in release it will likely cause a segfault.
   *
   * \param other The other node to move from.
   * \return A reference to the new node.
   */
  auto operator=(Node&& other) noexcept -> Node& {
    if (this == &other) {
      return *this;
    }

    debug_assert_node_valid(other);
    executor_   = other.executor_;
    info_       = other.info_;
    incoming_   = other.incoming_;
    successors_ = ripple_move(other.successors_);
    dependents_.store(
      other.dependents_.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    other.executor_ = nullptr;
    other.info_     = nullptr;
    other.incoming_ = 0;
    other.dependents_.store(0, std::memory_order_relaxed);
    return *this;
  }

  /*==--- [interface] ------ -----------------------------------------------==*/

  /**
   * Sets the executor of the node to use the given callable and its args.
   *
   * \note This will check that there is enough space in the node storage for
   *       the callable and its arguments. If there is not enough space, a
   *       compile time error is generated with the required additional number
   *       of bytes.
   *
   * \param  callable The callable object to store.
   * \param  args     The arguments to store.
   * \tparam F        The type of the callable object.
   * \tparam Args     The type of the arguments for the callable.
   */
  template <typename F, typename... Args>
  auto set_executor(F&& callable, Args&&... args) noexcept -> void {
    using Executor = NodeExecutorImpl<F, Args...>;
    if constexpr (size < sizeof(Executor)) {
      static_assert(
        Tuple<
          Num<size - sizeof(Executor)>,
          Num<size>,
          Num<sizeof(Executor)>,
          Num<sizeof(F)>,
          Tuple<Args, Num<sizeof(Args)>>...>::too_large,
        "Node storage is too small to allocate callable and its args!");
    }

    new (&storage_) Executor{ripple_forward(callable), ripple_forward(args)...};
    executor_ = reinterpret_cast<Executor*>(&storage_);
  }

  /**
   * Tries to run the node.
   * \return true if the node has no dependencies and therefore executes,
   *         otherwise returns false.
   */
  auto try_run() noexcept -> bool {
    if (dependents_.load(std::memory_order_relaxed) != size_t{0}) {
      return false;
    }

    /* Reset the node incase it needs to be run again, as well as to make sure
     * that it can't be run again until all dependents have run;
     *
     * \note This is very improtant that this is reset *before* the executor
     *       executes. If not, it's possible that multiple threads could see
     *       this node as having no dependencies, and both run the node.
     */
    dependents_.store(incoming_, std::memory_order_relaxed);
    executor_->execute();

    for (auto* successor : successors_) {
      if (successor && successor->num_dependents() > 0) {
        successor->dependents_.fetch_sub(1, std::memory_order_relaxed);
      }
    }
    return true;
  }

  /**
   * Add sthe given node as a successor for this node.
   * \param node The node to add as a successor to this node.
   */
  auto add_successor(Node& node) noexcept -> void {
    for (auto* successor : successors_) {
      if (successor == &node) {
        return;
      }
    }
    successors_.push_back(&node);
    node.increment_num_dependents();
  }

  /**
   * Adds the node with the given id as a friend of this node.
   *
   * \note A friend node is a node which can execute in parallel with another
   *       node, but which is used by the other node for the other node's
   *       operation, but is not modified by it.
   *
   *       For example, if x is a friend of y, then y *uses* x for an
   *       operation, and x cannot be modified by any operations on it until
   *       *y* has performed its operation.
   *
   * \param friend_id The id of the friend to add.
   */
  auto add_friend(typename NodeInfo::IdType friend_id) noexcept -> void {
    info_->friends.emplace_back(friend_id);
  }

  /**
   * Gets all friends for the node.
   * \return A container of all friend node ids.
   */
  auto friends() const noexcept -> typename NodeInfo::Friends& {
    return info_->friends;
  }

  /** Increments the number of dependents for the node. */
  auto increment_num_dependents() noexcept -> void {
    dependents_.fetch_add(1, std::memory_order_relaxed);
    incoming_ = dependents_.load(std::memory_order_relaxed);
  }

  /**
   * Gets the name of the node.
   * \return the name of the node.
   */
  auto name() const noexcept -> typename NodeInfo::Name {
    return info_ ? info_->name : NodeInfo::default_name;
  }

  /**
   * Gets the id of the node.
   * \return The id of the node.
   */
  auto id() const noexcept -> typename NodeInfo::IdType {
    return info_ ? info_->id : NodeInfo::default_id;
  }

  /**
   * Gets the kind of the node.
   * \return The kind of the node.
   */
  auto kind() const noexcept -> NodeKind {
    return info_ ? info_->kind : NodeKind::normal;
  }

  /**
   * Gets the execution target for the node.
   * \return The kind of the execution target for the node.
   */
  auto execution_kind() const noexcept -> ExecutionKind {
    return info_ ? info_->exec : default_execution_kind;
  }

  /**
   * Gets the number of dependents for the node.
   * \return The number of dependents for the node.
   */
  auto num_dependents() const noexcept -> CounterValue {
    return dependents_.load(std::memory_order_relaxed);
  }

 private:
  /*==--- [members] --------------------------------------------------------==*/

  Successors      successors_    = {};      //!< Array of successors.
  NodeExecutorPtr executor_      = nullptr; //!< The executor to run the node.
  NodeInfoPtr     info_          = nullptr; //!< Node's info.
  CounterValue    incoming_      = 0;       //!< Incomming connections.
  Counter         dependents_    = 0;       //!< Counter for dependents.
  char            storage_[size] = {};      //!< Additional storage.

  /**
   * Checks if the given node is valid.
   *
   * \note In release build this is disabled, while in debug builds
   *       this will terminate if the executor in node is a nullptr.
   *
   * \param node The node to check the validity of.
   */
  auto debug_assert_node_valid(const Node& node) const noexcept -> void {
    assert(node.executor_ != nullptr && "Node executor can't be a nullptr!");
  }
};

} // namespace ripple

#endif // RIPPLE_GRAPH_NODE_HPP
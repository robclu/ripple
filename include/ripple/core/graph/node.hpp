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
#include <ripple/core/math/math.hpp>
#include <ripple/core/utility/range.hpp>
#include <array>
#include <atomic>
#include <cassert>
#include <string>
#include <vector>

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// Forward declaration of the graph class.
class Graph;

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

//==--- [node info] ------------------------------------------------------==//

/// Defines the kinds of nodes.
enum class NodeKind : uint8_t {
  normal = 0, //!< Default kind of node.
  split  = 1, //!< Split kind of node.
  sync   = 2  //!< Node kind to explicity create a sync point.
};

/// NodeInfo implementation.
struct NodeInfo {
  // clang-format off
  /// Defines the type used for the node name.
  using name_t     = std::string;
  /// Defines the type used for the node id.
  using id_t       = uint64_t;
  /// Defines the type of the friends container.
  using friends_t  = std::vector<id_t>;

  /// Default id of the node.
  static constexpr id_t default_id   = std::numeric_limits<id_t>::max();
  /// Default name of the node.
  static constexpr auto default_name = "";
  // clang-format on

  //==--- [static] ---------------------------------------------------------==//

  /// Creates an id for a node from the indices, using hash combining.
  /// \param  indices The indices to create an id from.
  /// \tparam Size    The number of ids.
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

  /// Creates a name for a node from the indices.
  /// \param  indices The indices to create a name from.
  /// \tparam Size    The number of ids.
  template <size_t Size>
  static auto
  name_from_indices(const std::array<uint32_t, Size>& indices) -> std::string {
    std::string name = Size == 0 ? "" : std::to_string(indices[0]);
    for (auto i : range(Size - 1)) {
      name += "_" + std::to_string(indices[i + 1]);
    }
    return name;
  }

  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor for node info.
  NodeInfo() = default;

  /// Constructor for node info which moves \p n into this info's name.
  /// \param name The name of the node.
  NodeInfo(name_t name_) : name{std::move(name_)} {}

  /// Constructor to set the node name and id.
  /// \param name_ The name of the node.
  /// \param id_   The id of the node.
  NodeInfo(name_t name_, id_t id_) : name{std::move(name_)}, id{id_} {}

  /// Constructor to set the node name , id, and kind.
  /// \param name_ The name of the node.
  /// \param id_   The id of the node.
  NodeInfo(name_t name_, id_t id_, NodeKind kind_)
  : name{std::move(name_)}, id{id_}, kind{kind_} {}

  //==--- [deleted] --------------------------------------------------------==//

  // clang-format off
  /// Copy constructor -- deleted.
  NodeInfo(const NodeInfo& other) = default; 
  /// Move constructor deleted.
  NodeInfo(NodeInfo&& other)      = default;

  /// Move assignment -- deleted.
  auto operator=(const NodeInfo&) = delete;
  /// Copy assignment -- deleted;
  auto operator=(NodeInfo&&)      = delete;
  // clang-format on

  //==--- [members] --------------------------------------------------------==//

  name_t    name    = default_name;     //!< The name of the node.
  friends_t friends = {};               //!< Siblings for the node.
  id_t      id      = default_id;       //!< Id of the node.
  NodeKind  kind    = NodeKind::normal; //!< The kind of the node.
};

//==--- [node impl] --------------------------------------------------------==//

/// Implementation type of a node in a graph.
///
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
template <size_t Alignment>
class alignas(Alignment) Node {
  /// Allow the graph class access to the node for building the graph.
  friend Graph;

  // clang-format off
  /// We use a vector for the successors, because it's only 24 bytes, and
  /// doesn't limit the number of successors of a node. It only really matters
  /// that when all the successors are required, that the pointers are all
  /// contiguous so that they can be iterated over quickly.
  using successors_t  = std::vector<Node*>;
  /// Defines the value type of the counter.
  using counter_val_t = uint32_t;
  /// Defines the type of the unfinished counter.
  using counter_t     = std::atomic<counter_val_t>;
  /// Defines the type of the node information.
  using info_t        = NodeInfo;
  /// Defines the type of the pointer to the node information.
  using info_ptr_t    = info_t*;

  // clang-format off
  /// Memory required for the parent and the successors.
  static constexpr size_t node_mem_size = 
    sizeof(Node*) + sizeof(info_ptr_t) + sizeof(successors_t);
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

  /// Defines the size of the buffer for the storage.
  static constexpr size_t size =
    spare_mem + (spare_mem >= min_spare_mem ? size_t{0} : Alignment);

 public:
  //==--- Con/destruction --------------------------------------------------==//

  // clang-format off
  /// Default constructor -- uses the default values of the Node.
  Node() noexcept  = default;
  /// Default destructor.
  ~Node() noexcept {
    printf("Destroying : %3lu %s %p\n", 
      id(), name().c_str(), this);
  }
  // clang-format on

  /// Copy constructor which clones the executor and copies the rest of the node
  /// state.
  /// This will check against the executor being a nullptr in debug mode.
  ///
  /// \note This could be expensive if the node has a lot of successors.
  ///
  /// \note This does not copy the node's information. Information for a node
  ///       is unique and should be allocated and set after copying a noce.
  ///
  /// \param other The other task to copy from.
  Node(const Node& other) noexcept {
    debug_assert_node_valid(other);
    _executor = other._executor->clone(&_storage);
    _incoming = other._incoming;
    _dependents.store(
      other._dependents.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    for (size_t i = 0; i < other._successors.size(); ++i) {
      _successors[i] = other._successors[i];
    }
  }

  /// Move constructor which just calls the copy constructor, and invalidates
  /// the \p other node.
  /// \param other The other task to move from.
  Node(Node&& other) noexcept
  : _executor(other._executor),
    _info(other._info),
    _incoming(other._incoming),
    _successors(std::move(other._successors)),
    _dependents(other._dependents.load(std::memory_order_relaxed)) {
    other._executor = nullptr;
    other._info     = nullptr;
    other._incoming = 0;
    other._dependents.store(0, std::memory_order_relaxed);
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
  ///
  /// \note This does not copy the node information, which should be allocated
  ///       and then set after copying the node.
  ///
  /// \param other The other node to copy.
  auto operator=(const Node& other) noexcept -> Node& {
    if (this == &other) {
      return *this;
    }

    debug_assert_valid_node(other);
    _executor = other._executor->clone(&_storage);
    _incoming = other._incoming;
    _dependents.store(
      other._dependents.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    for (int i = 0; i < other._successors.size(); ++i) {
      _successors[i] = other._successors[i];
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
    _executor   = other._executor;
    _info       = other._info;
    _incoming   = other._incoming;
    _successors = std::move(other._successors);
    _dependents.store(
      other._dependents.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

    other._executor = nullptr;
    other._info     = nullptr;
    other._incoming = 0;
    other._dependents.store(0, std::memory_order_relaxed);
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

    new (&_storage)
      executor_t(std::forward<F>(callable), std::forward<Args>(args)...);
    _executor = reinterpret_cast<executor_t*>(&_storage);
  }

  /// Tries to run the node, returning true if the node's dependencies are met
  /// and the node runs, otherwise it returns false.
  auto try_run() noexcept -> bool {
    if (_dependents.load(std::memory_order_relaxed) != size_t{0}) {
      return false;
    }

    _executor->execute();

    // Reset the node incase it needs to be run again, as well as to make sure
    // that it can't be run again until all dependents have run;
    _dependents.store(_incoming, std::memory_order_relaxed);

    for (auto* successor : _successors) {
      if (successor) {
        successor->_dependents.fetch_sub(1, std::memory_order_relaxed);
      }
    }
    return true;
  }

  /// Add the \p node as a successor for this node.
  /// \param nod The node to add as a successor to this node.
  auto add_successor(Node& node) noexcept -> void {
    for (auto* successor : _successors) {
      if (successor == &node) {
        return;
      }
    }
    _successors.push_back(&node);
    node.increment_num_dependents();
  }

  /// Adds the node with \p id as a friend of the node. A friend is a node
  /// which can execute in parallel with another node, but which is used by the
  /// other node for the other node's operation, but it not modifies by it. For
  /// example, if x is a friend of y, then y _uses_ x for an operation, and x
  /// cannot be modified by any operations on it until _y_ has performed its
  /// operation.
  /// \param friend_id The id of the friend to add.
  auto add_friend(typename info_t::id_t friend_id) noexcept -> void {
    _info->friends.emplace_back(friend_id);
  }

  /// Returns a container of all node friend ids.
  auto friends() const noexcept -> typename info_t::friends_t& {
    return _info->friends;
  }

  /// Increments the number of dependents for the node.
  auto increment_num_dependents() noexcept -> void {
    _dependents.fetch_add(1, std::memory_order_relaxed);
    _incoming = _dependents.load(std::memory_order_relaxed);
  }

  /// Returns the name of the node.
  auto name() const -> typename info_t::name_t {
    return _info ? _info->name : info_t::default_name;
  }

  /// Returns the id of the node.
  auto id() const noexcept -> typename info_t::id_t {
    return _info ? _info->id : info_t::default_id;
  }

  /// Returns the kind of the node.
  auto kind() const noexcept -> NodeKind {
    return _info ? _info->kind : NodeKind::normal;
  }

  /// Returns the number of dependents.
  auto num_dependents() const noexcept -> counter_val_t {
    return _dependents.load(std::memory_order_relaxed);
  }

 private:
  //==--- [members] --------------------------------------------------------==//

  successors_t  _successors    = {};      //!< Array of successors.
  NodeExecutor* _executor      = nullptr; //!< The executor to run the node.
  info_ptr_t    _info          = nullptr; //!< Node's info.
  counter_val_t _incoming      = 0;       //!< Number of incomming connections.
  counter_t     _dependents    = 0;       //!< Counter for dependents.
  char          _storage[size] = {};      //!< Additional storage.

  //==--- [methods] --------------------------------------------------------==//

  /// Checks if the \p node is valid. In release mode this is disabled. In
  /// debug mode this will terminate if the executor in \p node is a nullptr.
  /// \param node  The node to check the validity of.
  auto debug_assert_node_valid(const Node& node) const noexcept -> void {
    assert(node._executor != nullptr && "Node executor can't be a nullptr!");
  }
};

} // namespace ripple

#endif // RIPPLE_GRAPH_NODE_HPP
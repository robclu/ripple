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

} // namespace ripple

#endif // RIPPLE_GRAPH_NODE_HPP
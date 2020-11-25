//==--- ripple/core/graph/detail/splitter_utils_.hpp ------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  splitter_utils_.hpp
/// \brief This file implements utilities for a splitter implementation.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_GRAPH_DETAIL_SPLITTER_UTILS__HPP
#define RIPPLE_GRAPH_DETAIL_SPLITTER_UTILS__HPP

#include "../node.hpp"
#include <ripple/core/container/block.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>

namespace ripple::detail {

/*==--- [dimensions] -------------------------------------------------------==*/

/**
 * Gets the number of dimensions if the type T is a block.
 * \tparam T The type to get the number of dimensions from.
 */
template <typename T>
constexpr size_t dims_from_block =
  any_block_traits_t<typename modification_traits_t<T>::Value>::dimensions;

/*==--- [iterator deref] ---------------------------------------------------==*/

/**
 * Gets a reference to the iterated type.
 *
 * \note This overlaod is only enabled for iterator types.
 *
 * \param iter The iterator to get the dereferences type from.
 * \return A reference to the iterated data.
 */
template <typename T, iterator_enable_t<T> = 0>
auto deref_if_iter(T&& iter) noexcept -> decltype(*iter)& {
  return *iter;
}

/**
 * Gets a reference to the iterated type.
 *
 * \note This overlaod is only enabled for non-iterator types and forwards the
 *       argument back.
 *
 * \param iter The iterator to get the dereferences type from.
 * \return A reference to the iterated data.
 */
template <typename T, non_iterator_enable_t<T> = 0>
decltype(auto) deref_if_iter(T&& t) noexcept {
  return ripple_forward(t);
}

/*==--- [padding] ----------------------------------------------------------==*/

/**
 * Copies the padding for the faces of the block iterated over by the iterato
 * for each of the dimensions which can be iterated over.
 *
 * \note This will fail at compile time if the iterator does not iterator over a
 *       Block.
 *
 * \param  exec_kind The kind of the execution for the operation.
 * \param  it        The iterator to copy the face padding for.
 * \tparam Iterator  The type of the iterator.
 */
template <typename Iterator>
auto copy_face_padding(ExecutionKind exec_kind, Iterator&& it) noexcept
  -> void {
  static_assert(
    is_iterator_v<Iterator>, "Friend node adding requires iterator!");
  static_assert(
    is_multiblock_v<decltype(*it)>, "Iterator must be over block type!");
  unrolled_for<iterator_traits_t<Iterator>::dimensions>([&](auto dim) {
    if (!it->first_in_dim(dim)) {
      constexpr auto specifier = CopySpecifier<dim, FaceLocation::start>();
      it->fill_padding(*it.offset(dim, -1), specifier, exec_kind);
    }
    if (!it->last_in_dim(dim)) {
      constexpr auto specifier = CopySpecifier<dim, FaceLocation::end>();
      it->fill_padding(*it.offset(dim, 1), specifier, exec_kind);
    }
  });
}

/*==--- [node operations] --------------------------------------------------==*/

/**
 * Adds all the nodes which would surround the node as friends of the node,
 * which ensures that any subsequent operations on the node will not happen
 * until the friends have finished.
 *
 * \param  node     The node to set the friends for.
 * \param  it       The iterator to the block for the node.
 * \tparam Align    The alignment of the node.
 * \tparam Iterator The type of the iterator.
 */
template <size_t Align, typename Iterator>
auto add_friend_nodes(Node<Align>& node, Iterator&& it) noexcept -> void {
  static_assert(
    is_iterator_v<Iterator>, "Friend node adding requires iterator!");
  static_assert(
    is_multiblock_v<decltype(*it)>, "Iterator must be over block type!");
  auto ids = it->indices;
  unrolled_for<iterator_traits_t<Iterator>::dimensions>([&](auto dim) {
    if (!it->first_in_dim(dim)) {
      ids[dim] -= 1;
      node.add_friend(NodeInfo::id_from_indices(ids));
      ids[dim] += 1;
    }

    if (!it->last_in_dim(dim)) {
      ids[dim] += 1;
      node.add_friend(NodeInfo::id_from_indices(ids));
      ids[dim] -= 1;
    }
  });
}

/**
 * Adds nodes for padding operations to the graph based off the iterator.
 * If the is_exclusive parameter is true, then friend nodes are added to the
 * resulting node in the graph so that subsequent operations on the new node
 * wont run until the friend nodes in the same level finish (i.e so that this
 * node--the friend-- finishes its copy bofore any other nodes run, so it make
 * a  stricter dependency in the graph).
 *
 * \note If the iterator iterates over a non-block enabled type then no node is
 *       added to the graph.
 *
 * \param  graph        The graph to add the node to.
 * \param  exec_kind    The kind of execution for the node.
 * \param  it           The iterator to add a node for.
 * \param  is_exclusive If data requires exclusive access (i.e is written to).
 * \tparam GraphType    The type of the graph.
 * \tparam Iterator     The type of the iterator.
 */
template <typename GraphType, typename Iterator>
auto add_padding_op_nodes_for_iter(
  GraphType&    graph,
  ExecutionKind exec_kind,
  Iterator&&    it,
  bool          is_exclusive) noexcept -> void {
  using IterValue = std::decay_t<decltype(*it)>;
  if constexpr (is_multiblock_v<IterValue>) {
    if (it->padding()) {
      const auto name = NodeInfo::name_from_indices(it->indices);
      const auto id   = NodeInfo::id_from_indices(it->indices);
      graph.emplace_named(
        NodeInfo(name, id, NodeKind::split, exec_kind),
        [exec_kind](auto&& iter) {
          copy_face_padding(exec_kind, ripple_forward(iter));
        },
        ripple_forward(it));

      /* See function comment. */
      if (is_exclusive) {
        auto* node = graph.find(name).value();
        detail::add_friend_nodes(*node, it);
      }
    }
  }
}

/**
 * Implementation of padding node addition.
 *
 * \note This overload is for iterator types and adds a padding node to the
 *       graph if the Modifier is a modifier.
 *
 * \param  graph      The graph to add the nodes to.
 * \param  exec_kind  The kind of execution for the nodes.
 * \param  arg        The argument to generate a padding node for.
 * \tparam Modifier   Modifier for how the argument data is accessed.
 * \tparam GraphType  The type of the graph.
 * \tparam Arg        The type of the arg.
 */
template <
  typename Modifier,
  typename GraphType,
  typename Arg,
  iterator_enable_t<Arg> = 0>
auto add_padding_op_nodes_impl(
  GraphType& graph, ExecutionKind exec_kind, Arg&& arg) noexcept -> void {
  using Traits = modification_traits_t<Modifier>;
  if constexpr (Traits::is_modifier) {
    add_padding_op_nodes_for_iter(
      graph, exec_kind, ripple_forward(arg), Traits::is_exclusive);
  }
}

/**
 * Implementation of padding node addition.
 *
 * \note This overload is for non-iterator types and does nothing.
 *
 * \param  graph      The graph to add the nodes to.
 * \param  exec_kind  The kind of execution for the nodes.
 * \param  arg        The argument to generate a padding node for.
 * \tparam Modifier   Modifier for how the argument data is accessed.
 * \tparam GraphType  The type of the graph.
 * \tparam Arg        The type of the arg.
 */
template <
  typename Modifier,
  typename GraphType,
  typename Arg,
  non_iterator_enable_t<Arg> = 0>
auto add_padding_op_nodes_impl(
  GraphType& graph, ExecutionKind exec_kind, Arg&& arg) noexcept -> void {}

// clang-format off
/**
 * Implementation of adding padding operations which expands the arguments into
 * the implementation function for the padding, adding padding nodes for any
 * arguments which are blocks are require padding based on the modifier for the
 * argument.
 *
 * \param  graph      The graph to add the nodes to.
 * \param  exec_kind  The kind of execution for the nodes.
 * \param  args       The arguments to generate padding nodes from.
 * \tparam Modifiers  Modifiers for how data is accessed.
 * \tparam GraphType  The type of the graph.
 * \tparam I          The indicies to access the modifiers for the args.
 * \tparam Args       The type of the args.
 */
template <typename Modifiers, typename GraphType, size_t... I, typename... Args>
auto add_padding_op_nodes_expanded(
  GraphType&               graph,
  ExecutionKind            exec_kind,
  std::index_sequence<I...>,
  Args&&...                args) noexcept -> void {
  // clang-format on
  (add_padding_op_nodes_impl<tuple_element_t<I, Modifiers>>(
     graph, exec_kind, ripple_forward(args)),
   ...);
}

/**
 * Adds padding operation nodes to the graph based on the modifiers.
 * \param  graph      The graph to add the nodes to.
 * \param  exec_kind  The kind of execution for the nodes.
 * \param  args       The arguments to generate padding nodes from.
 * \tparam Modifiers  Modifiers for how data is accessed.
 * \tparam GraphType  The type of the graph.
 * \tparam Args       The type of the args.
 */
template <typename Modifiers, typename GraphType, typename... Args>
auto add_padding_op_nodes(
  GraphType& graph, ExecutionKind exec_kind, Args&&... args) noexcept -> void {
  add_padding_op_nodes_expanded<Modifiers>(
    graph,
    exec_kind,
    std::make_index_sequence<sizeof...(Args)>(),
    ripple_forward(args)...);
}

} // namespace ripple::detail

#endif // RIPPLE_GRAPH_DETAIL_SPLITTER_UTILS__HPP
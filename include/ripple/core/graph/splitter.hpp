/**==--- ripple/core/graph/splitter.hpp -------------------- -*- C++ -*- ---==**
 *
 *                                 Ripple
 *
 *                  Copyright (c) 2019 - 2021 Rob Clucas
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  splitter.hpp
 * \brief This file implements functionality for splitting a tensor in a graph.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_GRAPH_SPLITTER_HPP
#define RIPPLE_GRAPH_SPLITTER_HPP

#include "modifier.hpp"
#include "detail/utils_.hpp"
#include "../container/block_extractor.hpp"
#include "../execution/execution_traits.hpp"
#include "../functional/invoke.hpp"

namespace ripple {

/*==--- [modifier application] ---------------------------------------------==*/

/**
 * Applies the properties of the modifier and dereferences the argument if it is
 * and iterator.
 *
 * \note This overload is only enabled if the modifier specifies shared memory
 *        use.
 *
 * \param  arg Argument to dereference and apply the shared memory modifer.
 * \tparam Mod The type of the modifier.
 * \tparam Arg The type of the argument.
 * \return A deferenced type with shared memory modification properties.
 */
template <typename Mod, typename Arg, shared_mod_enable_t<Mod> = 0>
decltype(auto)
apply_modifier_after_deref(Arg&& arg, ExpansionParams params) noexcept {
  return as_shared(
    detail::deref_if_iter(ripple_forward(arg)),
    detail::padding_if_iter(ripple_forward(arg)),
    params);
}

/**
 * Applies the properties of the modifier and dereferences the argument if it is
 * and iterator.
 *
 * \note This overload is only enabled if the modifier does not specify
 *       shared memory use. It therefore does not apply shared memory properties
 *       after dereferencing.
 *
 * \param  arg Argument to dereference and apply the shared memory modifer.
 * \tparam Mod The type of the modifier.
 * \tparam Arg The type of the argument.
 * \return A deferenced argument if it is an iterator.
 */
template <typename Mod, typename Arg, non_shared_mod_enable_t<Mod> = 0>
decltype(auto)
apply_modifier_after_deref(Arg&& arg, ExpansionParams params) noexcept {
  if constexpr (is_expander_modifier_v<Mod>) {
    return as_expansion(detail::deref_if_iter(ripple_forward(arg)), params);
  } else {
    return detail::deref_if_iter(ripple_forward(arg));
  }
}

/*==--- [fill indices implementation] --------------------------------------==*/

/**
 * Fills the given indices container if the iterator is an iterator.
 *
 * \note This overload is for iterators.
 *
 * \param  indices  The indices to fill from the iterator.
 * \param  set      If the indices have previously been set.
 * \param  it       The iterator to use to set the indices.
 * \tparam Size     The size of the indices container.
 * \tparam Iterator The type of the iterator.
 */
template <size_t Size, typename Iterator, iterator_enable_t<Iterator> = 0>
auto fill_indices(
  std::array<uint32_t, Size>& indices, bool& set, Iterator&& it) noexcept
  -> void {
  static_assert(
    iterator_traits_t<Iterator>::dimensions >= Size,
    "Iterator does not have enough dimensions to fill indices!");
  if (set) {
    return;
  }

  set = true;
  unrolled_for<Size>([&](auto i) { indices[i] = it->indices[i]; });
}

/**
 * Fills the given indices container if the iterator is an iterator.
 *
 * \note This overload is for non-iterators, and does nothing.
 *
 * \param  indices  The indices to fill from the iterator.
 * \param  set      If the indices have previously been set.
 * \param  it       The iterator to use to set the indices.
 * \tparam Size     The size of the indices container.
 * \tparam Iterator The type of the iterator.
 *
 */
template <size_t Size, typename Iterator, non_iterator_enable_t<Iterator> = 0>
auto fill_indices(
  std::array<uint32_t, Size>& indices, bool& set, Iterator&& it) noexcept
  -> void {}

/*==--- [splitter implementation] ------------------------------------------==*/

/**
 * This class splits a tensor by its partitions, creating a node in the graph
 * for each of the partitions in the tensor. It also determines the dependencies
 * between nodes in the graph, and adds memory transfer nodes for copying tensor
 * padding data, if required.
 */
struct Splitter {
 private:
  // clang-format off
  /**
   * Adds a node to the graph with the given name and callable by extracting
   * the types from the arguments.
   * \param  graph     The graph to add the node to.
   * \param  exe       The execution kind for the node.
   * \param  name      The name of the node to be added.
   * \param  id        The id of the name to be added.
   * \param  functor   The functor for the node.
   * \param  args      The args for the callable.
   * \tparam Mods      Modifers for how args are accessed in the callable.
   * \tparam Grapph    The type of the graph to add the node to.
   * \tparam F         The type of the functor.
   * \tparam Args      The types of the args.
   * \tparam I         The indices for modifier extraction.
   */
  template <
    typename Mods, typename Graph, typename F, typename... Args, size_t... I>
  static auto add_node(
    Graph&                    graph,
    ExecutionKind             exe,
    std::string               name,
    size_t                    id,
    F&&                       functor,
    std::index_sequence<I...>,
    std::array<ExpansionParams, sizeof...(Args)>& padding_mods,
    Args&&...                 args) noexcept -> void {
    // clang-format on
    graph.emplace_named(
      NodeInfo(name, id, NodeKind::split, exe),
      [&functor, exe](auto&&... node_args) {
        invoke_generic(
          exe, ripple_forward(functor), ripple_forward(node_args)...);
      },
      apply_modifier_after_deref<tuple_element_t<I, Mods>>(
        ripple_forward(args), padding_mods[I])...);
  }

 public:
  /**
   * Creates a split for any argument which is a tensor by creating a node
   * per partition in the tensor. If any of the arguments have modifiers then
   * the dependencies assosciated with the modifiers are added when adding the
   * nodes, as well as any additional nodes (such as memcpy nodes for padding).
   *
   * \param  graph   The graph to add the split nodes to.
   * \param  functor The functor which defines the operations on the nodes.
   * \param  args    The arguments for the functor.
   * \tparam Graph   The type of the graph.
   * \tparam F       The type of the functor.
   * \tparam Args    The type of the arguments.
   */
  template <typename Graph, typename F, typename... Args>
  static auto
  split(Graph& graph, ExecutionKind exe, F&& functor, Args&&... args) noexcept
    -> void {
    split_impl(
      graph,
      exe,
      ripple_forward(functor),
      BlockExtractor::extract_blocks_if_tensor(ripple_forward(args))...);
  }

 private:
  /**
   * Implementation of the split function. This first adds any padding nodes,
   * then it adds the computation nodes.
   *
   * \param graph The graph to add the nodes to.
   * \param functor The functor for the node operations.
   * \param args    The arguments for the functor.
   * \tparam Graph  The type of the graph.
   * \tparam F      The type of the functor.
   * \tparam Args   The type of the args.
   */
  template <typename Graph, typename F, typename... Args>
  static auto split_impl(
    Graph& graph, ExecutionKind exe, F&& functor, Args&&... args) noexcept
    -> void {
    constexpr size_t dimensions = max_element(detail::dims_from_block<Args>...);
    using Modifiers             = Tuple<std::decay_t<Args>...>;
    using Indices               = std::array<uint32_t, dimensions>;
    using PaddingMods           = std::array<ExpansionParams, sizeof...(Args)>;

    /* If any argument has a modifier, then padding nodes are needed, so add the
     * them for any tensor which has the modifier and multiple partitions. */
    if constexpr (has_modifier_v<Args...>) {
      invoke_generic(
        CpuExecutor(),
        [&](auto&&... unwrapped_args) {
          detail::add_padding_op_nodes<Modifiers>(
            graph, exe, ripple_forward(unwrapped_args)...);
        },
        unwrap_modifiers(ripple_forward(args))...);

      // Start new layer in the graph.
      graph.split_ids_.emplace_back(graph.nodes_.size());
    }

    PaddingMods padding_mods{
      get_modifier_expansion_params(ripple_forward(args))...};

    /* Add the nodes to perform the actual computation. */
    invoke_generic(
      CpuExecutor(),
      [&](PaddingMods& padding_mods, auto&&... unwrapped_args) {
        Indices indices;
        bool    set = false;
        (fill_indices(indices, set, unwrapped_args), ...);

        /* Emplace a node onto the graph which is the functor and the
         * args, converting any of the iteraors over tensor blocks into the
         * actual block that the operation will be performed on.
         */
        add_node<Modifiers>(
          graph,
          exe,
          NodeInfo::name_from_indices(indices),
          NodeInfo::id_from_indices(indices),
          ripple_forward(functor),
          std::make_index_sequence<sizeof...(Args)>(),
          padding_mods,
          ripple_forward(unwrapped_args)...);
      },
      padding_mods,
      unwrap_modifiers(ripple_forward(args))...);
  }
};

} // namespace ripple

#endif // RIPPLE_GRAPH_SPLITTER_HPP

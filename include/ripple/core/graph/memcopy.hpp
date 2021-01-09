/**==--- ripple/core/graph/memcpy.hpp ---------------------- -*- C++ -*- ---==**
 *
 *                                 Ripple
 *
 *                  Copyright (c) 2019 - 2021 Rob Clucas
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  memcopy.hpp
 * \brief This file implements functionality for copying padding data between
 *        partitions of a tensor and adding the necessary operations to a
 *        graph.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_GRAPH_MEMCOPY_HPP
#define RIPPLE_GRAPH_MEMCOPY_HPP

#include "detail/utils_.hpp"
#include "../container/block_extractor.hpp"
#include "../functional/invoke.hpp"

namespace ripple {

/**
 * This class finds partitions in a tensor, and if so, appends operations onto
 * a graph which perform data transer such that the padding data of each
 * partition is correctly filled with internal data from neighbour partitions.
 */
struct Memcopy {
 public:
  /**
   * Adds neceasarry memcopy nodes if any of the arguments are tensors, creating
   * any necessary dependencies due to modifiers on the tensors.
   *
   * \param  graph   The graph to add the memcopy nodes to.
   * \param  exe     The kind of the execution for the operations.
   * \param  args    The arguments for the functor.
   * \tparam Graph   The type of the graph.
   * \tparam Args    The type of the arguments.
   */
  template <typename Graph, typename... Args>
  static auto
  memcopy(Graph& graph, ExecutionKind exe, Args&&... args) noexcept -> void {
    memcopy_impl(
      graph,
      exe,
      BlockExtractor::extract_blocks_if_tensor(ripple_forward(args))...);
  }

 private:
  /**
   * Implementation of the memcopy function. T
   *
   * \param  graph  The graph to add the nodes to.
   * \param  exe    The kind of the execution for the operations.
   * \param  args   The arguments for the functor.
   * \tparam Graph  The type of the graph.
   * \tparam Args   The type of the args.
   */
  template <typename Graph, typename... Args>
  static auto
  memcopy_impl(Graph& graph, ExecutionKind exe, Args&&... args) noexcept
    -> void {
    using Modifiers = Tuple<std::decay_t<Args>...>;

    /* If any argument has a modifier, then padding nodes are needed, so add the
     * them for any tensor which has the modifier and multiple partitions. */
    invoke_generic(
      CpuExecutor(),
      [&](auto&&... unwrapped_args) {
        detail::add_padding_op_nodes<Modifiers>(
          graph, exe, ripple_forward(unwrapped_args)...);
      },
      unwrap_modifiers(ripple_forward(args))...);
  }
};

} // namespace ripple

#endif // RIPPLE_GRAPH_MEMCOPY_HPP
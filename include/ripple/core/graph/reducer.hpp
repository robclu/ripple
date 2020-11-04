//==--- ripple/core/graph/reducer.hpp ---------------------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reducer.hpp
/// \brief This file implements functionality for reducing tensor data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_GRAPH_REDUCER_HPP
#define RIPPLE_GRAPH_REDUCER_HPP

#include <ripple/core/algorithm/reduce.hpp>
#include <ripple/core/container/block_extractor.hpp>

namespace ripple {

/**
 * This class holds the result of a reduction over data of type T. It stores
 * an atomic type T and a boolean which spcifies if the reduction is complete.
 * It is used to perform a reduction across multiple devices.
 *
 * \tparam T The type of the data for the reduction.
 */
template <typename T>
class ReductionResult {
 public:
  /**
   * Default constructor, set the results to be unfinished and the value to be
   * zero.
   */
  ReductionResult() noexcept = default;

  /**
   * Constructor to initialize the result to the given value.
   * \param value The value to set the result to.
   */
  ReductionResult(const T& value) noexcept : initial_{value} {
    set_value(value);
  }

  /**
   * Gets the result of the redution.
   *
   * \note If a call to `finished()` returns false, then the result of this
   *       call is undefined.
   *
   * \return The value of the reduction result.
   */
  auto value() const noexcept -> T {
    return value_.load(std::memory_order_relaxed);
  }

  /**
   * Gets the result of the redution and then resets the value of the result to
   * be the initial value.
   *
   * \note If a call to `finished()` returns false, then the result of this
   *       call is undefined.
   *
   * \return The value of the reduction result.
   */
  auto value_and_reset() noexcept -> T {
    const T val = value_.load(std::memory_order_relaxed);
    reset();
    return val;
  }

  /**
   * Resets the reduction result to the initial value, and for the refuction
   * result to not be finished.
   */
  auto reset() noexcept -> void {
    set_value(initial_);
    set_unfinished();
  }

  /**
   * Sets the result to the given value.
   * \param value The value to set the result to.
   */
  auto set_value(const T& value) noexcept -> void {
    value_.store(value, std::memory_order_relaxed);
  }

  /**
   * Determines if the reduction is finished and hence if the value is valid.
   * \return true if threduction is finished and the value is valid.
   */
  auto finished() const noexcept -> bool {
    return finished_.load(std::memory_order_relaxed);
  }

  /**
   * Sets that the reduction is finished.
   */
  auto set_finished() noexcept -> void {
    finished_.store(true, std::memory_order_relaxed);
  }

  /**
   * Sets the the reduction is not finished.
   */
  auto set_unfinished() noexcept -> void {
    finished_.store(false, std::memory_order_relaxed);
  }

 private:
  std::atomic<T>    value_;           //!< The result of the reduction.
  T                 initial_{0};      //!< The initial value of the result.
  std::atomic<bool> finished_{false}; //!< If the reduction is finished.
};

/**
 * Creates a ReductionResult with the given intitial value, and which is not
 * finished with the reduction.
 *
 * \param  value The value to set the result to.
 * \tparam T     The type of the data for the result.
 * \return The initializes reduction result.
 */
template <typename T>
auto make_reduction_result(const T& value = T()) noexcept
  -> ReductionResult<T> {
  return ReductionResult<T>{value};
}

/**
 * This struct appends nodes to a graph which perform a reduction of tensor
 * data into a final result.
 *
 * It add a node for reduction per partition in the tensor, and a final node to
 * reduce all the partial nodes from the partitions. It sets a reduction result
 * to finished in the final node.
 */
struct Reducer {
 public:
  // clang-format off
  /**
   * Implementation of a reduction over the data by creating parallel nodes
   * for each of the blocks in the given tensor and adding them to the given
   * graph.
   *
   * The result for each node is then reduced into the result using the given
   * predicate.
   * 
   * \param  graph     The graph to append the reduction nodes to.
   * \param  exec_kind The kind of the execution for the reduction.
   * \param  data      The data to reduce.
   * \param  result    The result to reduce into.
   * \param  pred      The predicate which defines the reduction operation.
   * \param  args      Additional arguments for the predicate.
   * \tparam GraphType The type of the graph to append the nodes to.
   * \tparam T         The type of the data in the tensor.
   * \tparam Dims      The number of dimensions for the tensor.
   * \tparam Pred      The type of the predicate for the reduction.
   * \tparam Args      The types of the arguments for the predicate.
   */
  template <
    typename    GraphType,
    typename    T,
    size_t      Dims,
    typename    Pred,
    typename... Args>
  static auto reduce(
    GraphType&          graph,
    ExecutionKind       exec_kind,
    Tensor<T, Dims>&    data,
    ReductionResult<T>& result,
    Pred&&              pred,
    Args&&...           args) noexcept -> void {
    reduce_impl<Dims>(
      graph,
      exec_kind,
      BlockExtractor::extract_blocks_if_tensor(data),
      result,
      ripple_forward(pred),
      ripple_forward(args)...);
  }

 private:
  /**
   * Implementation of the reduction which adds a node for each block in the
   * tensor to the graph, and then adds the result to the reduction result.
   * 
   * \param  graph     The graph to append the reduction nodes to.
   * \param  exec_kind The execution kind for the nodes.
   * \param  blocks    The blocks to reduce.
   * \param  result    The result to reduce into.
   * \param  predicate The predicate which defines the reduction operation.
   * \param  args      Additional arguments for the predicate.
   * \tparam Dims      The number of dimensions to reduce over.
   * \tparam Graph     The type of the graph to append the nodes to.
   * \tparam Blocks    The type of the blocks to reduce.
   * \tparam T         The type of the data in the tensor.
   * \tparam Pred      The type of the predicate for the reduction.
   * \tparam Args      The types of the arguments for the predicate.
   */
  template <
    size_t      Dims,
    typename    Graph,
    typename    Blocks,
    typename    T,
    typename    Pred,
    typename... Args>
  static auto reduce_impl(
    Graph&              graph,
    ExecutionKind       exec_kind,
    Blocks&&            blocks,
    ReductionResult<T>& result,
    Pred&&              predicate,
    Args&&...           args) noexcept -> void {
    using Indices = std::array<uint32_t, Dims>;
    // clang-format on
    invoke_generic(
      CpuExecutor(),
      [&, exec_kind](auto&& iter) {
        Indices indices;
        bool    set = false;
        fill_indices(indices, set, iter);

        /* Emplace an operation into the graph which reduces the block pointed
         * to by the tensor block iterator, reducing the result from the block
         * reduction into the final result of the overall reduction across the
         * entire tensor. */
        const auto name = NodeInfo::name_from_indices(indices);
        const auto id   = NodeInfo::id_from_indices(indices);
        graph.emplace_named(
          NodeInfo(name, id, NodeKind::split, exec_kind),
          [&result, exec_kind](auto block_iter, auto&& pred, auto&&... as) {
            auto temp_result = block_iter->reduce(
              exec_kind, ripple_forward(pred), ripple_forward(as)...);
            temp_result =
              pred(result.value(), temp_result, ripple_forward(as)...);
            result.set_value(temp_result);
          },
          ripple_forward(iter),
          ripple_forward(predicate),
          ripple_forward(args)...);
      },
      ripple_forward(blocks));
  }
};

} // namespace ripple

#endif // RIPPLE_GRAPH_REDUCER_HPP
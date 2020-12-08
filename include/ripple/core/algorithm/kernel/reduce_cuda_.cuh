//==--- ripple/core/algorithm/kernel/reduce_cuda_.cuh ------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce_cuda_.cuh
/// \brief This file implements functionality to reduce a multi dimensional
///        block on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALGORITHM_KERNEL_INVOKE_CUDA__CUH
#define RIPPLE_ALGORITHM_KERNEL_INVOKE_CUDA__CUH

#include "reduce_cpp_.hpp"
#include "../../arch/gpu_utils.hpp"
#include "../../allocation/multiarch_allocator.hpp"
#include "../../execution/synchronize.hpp"
#include "../../execution/thread_index.hpp"
#include "../../functional/invoke.hpp"
#include "../../iterator/iterator_traits.hpp"
#include "../../utility/portability.hpp"

#if defined(ripple_gpu_compile)
  #include <cooperative_groups/reduce.h>
#endif

namespace ripple::kernel::gpu {
namespace detail {

/**
 * Performs a reduction of over all currently active threads where each thread
 * has the given value for the reduction. This sets the value based on the
 * predicate.
 *
 * \param  value The value to reduce.
 * \param  pred  The predicate for the reduction.
 * \tparam T     The type of the value.
 * \tparam Pred  The type of the predicate.
 * \return The result of the reduction.
 */
template <typename T, typename P>
ripple_device auto reduce_impl(T value, P&& pred) noexcept -> T {
#if defined(ripple_gpu_compile)
  namespace cg = cooperative_groups;
  auto active  = cg::coalesced_threads();
  T    result  = cg::reduce(active, value, ripple_forward(pred));
  active.sync();
  return result;
#else
  return T{0};
#endif
}

/**
 * Perfroms a block reduction of the data being iterated over, writing the
 * result for each block to the results iterator, which can then be reduced
 * further to get the final result.
 *
 * \param  it          The iterator over all data to reduce.
 * \param  results_it  The results iterator for each block.
 * \param  exec_params The parameters for the execution.
 * \param  pred        The predicate for the reduction.
 * \tparam Iterator    The type of the iterator over the reduction data.
 * \tparam ResIterator The type of the iterator for each block's results.
 * \tparam ExeImpl     The implementation of the execution param interface.
 * \tparam Pred        The type of the predicate.
 */
template <typename Iterator, typename ResIterator, typename Pred>
ripple_global auto
reduce_block(Iterator it, ResIterator results_it, Pred pred) -> void {
  constexpr size_t dims     = iterator_traits_t<Iterator>::dimensions;
  bool             in_range = true;
  unrolled_for<dims>([&](auto dim) {
    if (!it.is_valid(dim)) {
      in_range = false;
    }
  });

  if (!in_range) {
    return;
  }

  unrolled_for<dims>([&](auto dim) { it.shift(dim, global_idx(dim)); });
  auto result = *it;
  result      = reduce_impl(result, ripple_forward(pred));

  // Copy the results to the output:
  if (first_thread_in_block()) {
    unrolled_for<dims>(
      [&](auto dim) { results_it.shift(dim, block_idx(dim)); });
    *results_it = result;
  }
}

} // namespace detail

/**
 * Reduces the \p block using the \p pred.
 *
 * \note This uses the global host device allocator which significanlty
 *       improved performance.e
 *
 * \param  block     The block to invoke the callable on.
 * \param  pred      The predicate for the reduction.
 * \tparam T         The type of the data in the block.
 * \tparam Dims      The number of dimensions in the block.
 * \tparam Pred      The type of the predicate for the reduction.
 */
template <typename T, size_t Dims, typename Pred>
auto reduce(const DeviceBlock<T, Dims>& block, Pred&& pred) {
  using ExecParams       = default_shared_exec_params_t<Dims, T>;
  auto [threads, blocks] = get_exec_size(block, ExecParams{});

  /*
   * NOTE: The allocator here is important. Without it, when performing the
   * reduction across multiple devices, if the allocation is done with
   * cudaMalloc and malloc, rather than the allocator, even if small,
   * significantly reduces the performance of the reduction because of both
   * the synchronization required and allocation time.
   */
  ::ripple::gpu::set_device(block.device_id());
  DeviceBlock<T, Dims> results(block.stream(), &multiarch_allocator());
  results.set_device_id(block.device_id());

  // clang-format off
  unrolled_for<Dims>([&](auto d, auto& blocks) {
    results.resize_dim(d, 
      d == dimx() ? blocks.x : d == dimy() ? blocks.y : blocks.z);
  }, blocks);
  // clang-format on
  results.reallocate();

  ripple_if_cuda(detail::reduce_block<<<blocks, threads, 0, block.stream()>>>(
    block.begin(), results.begin(), pred));
  ::ripple::gpu::check_last_error();

  /* Reduce the results -- this will automatically sync with the previous
   * operation, since we use synchronous operations in the host. */
  const auto res = results.as_host(BlockOpKind::synchronous);
  return reduce(res, ripple_forward(pred));
}

} // namespace ripple::kernel::gpu

#endif // RIPPLE_ALGORITHM_KERNEL_REDUCE_CUDA__CUH

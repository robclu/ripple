//==--- ripple/algorithm/kernel/reduce_cuda_.cuh ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
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
#include <ripple/execution/synchronize.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/functional/invoke.hpp>
#include <ripple/iterator/iterator_traits.hpp>

namespace ripple::kernel::cuda {
namespace detail {

/// Performs a reduction on the block data iterated over by the \p it in the
/// dimension given by Dim. 
/// \param  it        An iterator over shared memory.
/// \param  elements  The total number of elements in the dimension.
/// \param  pred      The predicate for the reduction.
/// \param  args      Arguments for the callable.
/// \tparam Dim       The dimension to reduce over.
/// \tparam Iterator  The type of the iterator.
/// \tparam Pred      The type of the predicate.
/// \tparam Args      The type of the arguments.
template <size_t Dim, typename Iterator, typename Pred, typename... Args>
ripple_device_only auto reduce_block_for_dim(
  Iterator&& it, size_t elements, Pred&& pred, Args&&... args
) -> void {
#if defined(__CUDACC__)
  // Force dim to be contexpr:
  constexpr auto dim = Dim == 0 ? dim_x : Dim == 1 ? dim_y : dim_z;

  // Account for threads which may not run in last block of dimension:
  elements = std::min(
    it.size(dim), elements - block_idx(dim) * block_size(dim)
  );

  // Compute iteration, essentially log_2(elements) + 1;
  auto dim_size = elements; auto iters = 1;
  while (dim_size > 1) {
    iters++; dim_size >>= 1;
  }

  // Need to use for loop instead of while for syncthreads correctnes:
  // This previously used a while loop to avoid the iteration calculation above,
  // but __syncthreads in the while loop caused erros. The register usage an
  // performance of this is the same, however.
  //
  // It may be possible to use the while loop with coalesced_group.sync(). The
  // performance of these two options needs to be tested though. It's likely
  // that a reduction is almost never the bottleneck though, so perhaps not
  // worth the effort.
  auto left = it, right = it;
  for (auto i : range(iters)) {
    const auto rem   = elements & 1;
    elements       >>= 1;
    if (thread_idx(dim) < elements) {
      right = it.offset(dim, elements + rem);
      pred (left, right, args...);
    }
    elements += rem;
    sync_block();
  }
#endif // __CUDACC__
}

/// Perfroms a block reduction in shared memory over each of the dimensions,
/// putting the result into the \p results_it iterator for each block.
///
/// \param  it          The iterator over all data to reduce.
/// \param  results_it  The results iterator for each block.
/// \param  exec_params The parameters for the execution.
/// \param  pred        The predicate for the reduction.
/// \param  args        Arguments for the predicate.
/// \tparam Iterator    The type of the iterator over the reduction data.
/// \tparam ResIterator The type of the iterator for each block's results.
/// \tparam ExeImpl     The implementation of the execution param interface.
/// \tparam Pred        The type of the predicate.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    Iterator   ,
  typename    ResIterator,
  typename    ExecImpl   ,
  typename    Pred       ,
  typename... Args
>
ripple_global auto reduce_block_shared(
  Iterator    it         ,
  ResIterator results_it ,
  ExecImpl    exec_params,
  Pred        pred       ,
  Args...     args
) -> void {
#if defined(__CUDACC__)
  // Create the shared memory buffer and iterator over the data:
  constexpr auto dims       = it.dimensions();
  constexpr auto alloc_size = exec_params.template allocation_size<dims>();
  __shared__ char buffer[alloc_size];
  auto shared_it = 
    exec_params.template iterator<dims>(static_cast<void*>(buffer));

  // Offset the iterators to the global and thread indices:
  bool in_range  = true;
  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim    = d;
    const auto idx        = global_idx(dim);
    const auto must_shift = idx < it.size(dim) && in_range;
    if (must_shift) {
      it.shift(dim, idx);
      shared_it.shift(dim, thread_idx(dim) + shared_it.padding());
    } else {
      in_range = false;
    }
  });
  *shared_it = *it;
  sync_block();

  if (!in_range) {
    return;
  }

  // Reduce each dimension to get a single value per block:
  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = dims - 1 - d;
    reduce_block_for_dim<dim>(shared_it, it.size(dim), pred, args...);
  });  

  // Copy the results to the output:
  if (first_thread_in_block()) {
    unrolled_for<dims>([&] (auto d) {
      constexpr auto dim = d;
      results_it.shift(dim, block_idx(dim));
    });
    *results_it = *shared_it;
  }
#endif // __CUDACC__
}

} // namespace detail

/// Reduces the \p block using the \p pred.
///
/// \param  block     The block to invoke the callable on.
/// \param  pred      The predicate for the reduction.
/// \param  args      Arguments for the predicate.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Pred      The type of the predicate for the reduction.
/// \tparam Args      The type of the arguments for the invocation.
template <typename T, size_t Dims, typename Pred, typename... Args>
auto reduce(const DeviceBlock<T, Dims>& block, Pred&& pred, Args&&... args) {
#if defined(__CUDACC__)
  using exec_params_t    = default_shared_exec_params_t<Dims, T>;
  auto [threads, blocks] = get_exec_size(block, exec_params_t{});
  
  DeviceBlock<T, Dims> results;
  unrolled_for<Dims>(
    [&results] (auto d, auto& blocks) {
      constexpr auto dim = d;
      results.resize_dim(dim, 
        dim == dim_x ? blocks.x :
        dim == dim_y ? blocks.y :
        blocks.z
      );
    }, 
    blocks
  );
  results.reallocate();

  detail::reduce_block_shared<<<blocks, threads>>>(
    block.begin(), results.begin(), exec_params_t(), pred, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());

  // Reduce the results:
  const auto res = results.as_host();
  return reduce(res, std::forward<Pred>(pred), std::forward<Args>(args)...);
#endif // __CUDACC__
}

} // namespace ripple::kernel::cuda

#endif // RIPPLE_ALGORITHM_KERNEL_REDUCE_CUDA__CUH

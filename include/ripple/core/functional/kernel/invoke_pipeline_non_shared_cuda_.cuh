//==--- ../invoke_pipeline_non_shared_cuda_.cuh ------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_pipeline_non_shared_cuda_.cuh
/// \brief This file implements functionality to invoke a pipeline on various
///        container objects on the device, without shared memory.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_NON_SHARED_CUDA__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_NON_SHARED_CUDA__HPP

#include "invoke_utils_.cuh"
#include "../pipeline.hpp"
#include <ripple/core/container/device_block.hpp>
#include <ripple/core/execution/execution_params.hpp>
#include <ripple/core/execution/execution_size.hpp>
#include <ripple/core/execution/execution_traits.hpp>
#include <ripple/core/execution/synchronize.hpp>
#include <ripple/core/execution/detail/thread_index_impl_.hpp>
#include <ripple/core/execution/thread_index.hpp>

namespace ripple::kernel::cuda {
namespace detail {

//==--- [non-shared pipeline implementation] -------------------------------==//

/// Invokes each stage of the \p pipeline on the \p it iterator. 
///
/// This first offsets the iterator and then applies the pipeline to the
/// iterator.
///
/// Additionally, it offsets any of the \p args if they are iterators.
///
/// \param  it        The iterator to the global data.
/// \param  pipeline  The pipeline to invoke on the iterator.
/// \param  args      Additional arguments for the pipeline.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Iterator  The type of the iterator.
/// \tparam Ops...    The type of the pipeline operations.
/// \tparam Args...   The type of additional argumetns for the pipeline.
template <size_t Dims, typename Iterator, typename... Ops, typename... Args>
ripple_global auto invoke_non_shared_pipeline(
  Iterator it, Pipeline<Ops...> pipeline, Args... args
) -> void {
#if defined(__CUDACC__)
  unrolled_for<Dims>([&] (auto _dim) {
    constexpr auto dim = size_t{_dim};
    const     auto idx = global_idx(dim);
    ::ripple::detail::global_elements(dim) = it.size(dim);
  });

  if (!util::shift_in_range_global(it, args...)) {
    return;
  }

  // Run each stage of the pipeline ...
  using pipeline_t = Pipeline<Ops...>;
  unrolled_for<pipeline_t::num_stages>([&] (auto i) {
    constexpr auto stage = size_t{i};
    pipeline.template get_stage<stage>().operator()(it, args...);

    if constexpr (pipeline_t::num_stages > 0) {
      sync_block();
    }
  });
#endif // __CUDACC__
}

} // namespace detail

//==--- [variadic overload] ------------------------------------------------==//

/// Invokes the \p pipeline on each element in the \p block. If any of the \p
/// args are blocks, then this extracts the iterators from the block and
/// forwards those.
///
/// \param  block     The block to invoke the pipeline on.
/// \param  pipeline  The pipeline to invoke on the block.
/// \param  args      Additional arguements to the pipeline.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Ops       The operations in the pipeline.
/// \tparam Args      The type of additional arguments.
template <
  typename T, size_t Dims, typename... Ops, typename... Args,
  variadic_ge_enable_t<1, Args...> = 0
>
auto invoke_non_shared_pipeline(
  DeviceBlock<T, Dims>& block, const Pipeline<Ops...>& pipeline, Args&&... args
) -> void {
#if defined(__CUDACC__)
  auto exec_params       = default_exec_params_t<Dims>{};
  auto [threads, blocks] = get_exec_size(block, exec_params);

  detail::invoke_non_shared_pipeline<Dims><<<
    blocks, threads, 0, block.stream()
  >>>(
    block.begin(), pipeline, iter_or_ref(std::forward<Args>(args))...
  );
  ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
#endif
}

//==--- [non-variadic overload] --------------------------------------------==//

/// Invokes the \p pipeline on each element in the \p block
///
/// \param  block     The block to invoke the pipeline on.
/// \param  pipeline  The pipeline to invoke on the block.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Ops       The operations in the pipeline.
template <typename T, size_t Dims, typename... Ops>
auto invoke_non_shared_pipeline(
  DeviceBlock<T, Dims>& block, const Pipeline<Ops...>& pipeline
) -> void {
#if defined(__CUDACC__)
  auto exec_params       = default_exec_params_t<Dims>{};
  auto [threads, blocks] = get_exec_size(block, exec_params);

  detail::invoke_non_shared_pipeline<Dims><<<
    blocks, threads, 0, block.stream()
  >>>(
    block.begin(), pipeline
  );
  ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
#endif
}

} // namespace ripple::kernel::cuda

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_NON_SHARED_CUDA__HPP


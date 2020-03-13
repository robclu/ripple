//==--- ../kernel/invoke_pipeline_shared_cuda_.cuh --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_pipeline_shared_multi_cuda_.cuh
/// \brief This file implements functionality to invoke a pipeline on a block,
///        with any number of additional arguments.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_SHARED_CUDA__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_SHARED_CUDA__HPP

#include "invoke_utils_.cuh"
#include "../pipeline.hpp"
#include <ripple/core/boundary/copy_loader.hpp>
#include <ripple/core/container/device_block.hpp>
#include <ripple/core/execution/execution_params.hpp>
#include <ripple/core/execution/execution_size.hpp>
#include <ripple/core/execution/execution_traits.hpp>
#include <ripple/core/execution/dynamic_execution_params.hpp>
#include <ripple/core/execution/static_execution_params.hpp>

namespace ripple::kernel::cuda {
namespace detail {

//==--- [shared pipeline multi implementation] -----------------------------==//

/// Invokes each stage of the \p pipeline on the \p shared_it iterator. 
///
/// This first offsets both the iterators and then sets the data in the shared
/// iterator to be that of the global iterator.
///
/// It then loads the padding data for the shared iterator, if the iterator has
/// padding.
///
/// Each stage of the pipeline is then applied to the shared iterator, before
/// coping the updated shared iterator data to the global iterator.
///
/// \param  it             The iterator to the global data.
/// \param  shared_it      The iterator to the shared data.
/// \param  pipeline       The pipeline to invoke on the iterator.
/// \param  args           Additional arguments for the pipeline.
/// \tparam Iterator       The type of the iterator.
/// \tparam SharedIterator The type of the shared iterator.
/// \tparam Ops...         The type of the pipeline operations.
/// \tparam Args...        The type of additional argumetns for the pipeline.
template <
  typename Iterator, typename SharedIterator, typename... Ops, typename... Args
>
ripple_device_only auto invoke_shared_pipeline_multi(
  Iterator&&        it       , 
  SharedIterator&&  shared_it,
  Pipeline<Ops...>& pipeline ,
  Args&&...         args
) -> void {
#if defined(__CUDACC__)
  // Any thread out of the range of the iterator must return.
  if (!util::shift_in_range(it, shared_it, args...)) {
    return;
  }

  util::set_iter_data(it, shared_it);
  util::set_iter_boundary(it, shared_it);
  sync_block();

  // Run each stage of the pipeline ...
  using pipeline_t = Pipeline<Ops...>;
  unrolled_for<pipeline_t::num_stages>([&] (auto i) {
    constexpr auto stage = size_t{i};
    pipeline.template get_stage<stage>().operator()(
      shared_it, std::forward<Args>(args)...
    );

    if constexpr (pipeline_t::num_stages > 0) {
      sync_block();
    }
  });
  *it = *shared_it;
#endif // __CUDACC__
}

/// Invokes the \p pipeline on a shared memory iterator whose data is set from
/// the global iterator \p it. The shared memory for this function is allocated
/// statically, which has slightly better performance.
///
/// This function specifically creates the shared memory iterator, and then
/// calls the implementation which invokes each stage of the pipeline on the
/// iterator.
///
/// \param  it          The iterator to invoke the callable on.
/// \param  exec_params The parameters for the execution.
/// \param  pipeline    The pipeline to invoke on the iterator.
/// \param  args        Additional arguments for the pipeline.
/// \tparam Iterator    The type of the global iterator.
/// \tparam ExeImpl     The implementation of the execution param interface.
/// \tparam Ops...      The type of the pipeline operations.
/// \tparam Args...     The type of additional argumetns for the pipeline.
template <
  typename Iterator, typename ExecImpl, typename... Ops, typename... Args 
>
ripple_global auto invoke_static_shared_pipeline_multi(
  Iterator it, ExecImpl exec_params, Pipeline<Ops...> pipeline, Args... args
) -> void {
  constexpr auto dims       = it.dimensions();
  constexpr auto alloc_size = exec_params.template allocation_size<dims>();
  __shared__ char buffer[alloc_size];

  unrolled_for<dims>([&] (auto dim) {
    ::ripple::detail::global_elements(dim) = it.size(dim);
  });

  auto shared_it = 
    exec_params.template iterator<dims>(static_cast<void*>(buffer));

  invoke_shared_pipeline_multi(it, shared_it, pipeline, args...);
}

/// Invokes the \p pipeline on a shared memory iterator whose data is set from
/// the global iterator \p it. The shared memory for this function is allocated
/// dynamically.
///
/// This function specifically creates the shared memory iterator, and then
/// calls the implementation which invokes each stage of the pipeline on the
/// iterator.
///
/// \param  it          The iterator to invoke the callable on.
/// \param  exec_params The parameters for the execution.
/// \param  pipeline    The pipeline to invoke on the iterator.
/// \param  args        Additional arguments for the pipeline.
/// \tparam Iterator    The type of the global iterator.
/// \tparam ExeImpl     The implementation of the execution param interface.
/// \tparam Ops...      The type of the pipeline operations.
/// \tparam Args...     The type of additional argumetns for the pipeline.
template <
  typename Iterator, typename ExecImpl, typename... Ops, typename... Args
>
ripple_global auto invoke_dynamic_shared_pipeline_multi(
  Iterator it, ExecImpl exec_params, Pipeline<Ops...> pipeline, Args... args
) -> void {
  constexpr auto dims = it.dimensions();
  extern __shared__ char buffer[];

  unrolled_for<dims>([&] (auto dim) {
    ::ripple::detail::global_elements(dim) = it.size(dim);
  });

  auto shared_it = 
    exec_params.template iterator<dims>(static_cast<void*>(buffer));

  invoke_shared_pipeline_multi(it, shared_it, pipeline, args...);
}

} // namespace detail

//==--- [variadic overload] ------------------------------------------------==//

/// Invokes the \p pipeline on each element in the \p block, forwarding the \p
/// args to the pipeline, and offsetting any of the args to their global thread
/// indices if they are iterator types.
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
auto invoke_pipeline(
  DeviceBlock<T, Dims>& block, const Pipeline<Ops...>& pipeline, Args&&... args
) -> void {
#if defined(__CUDACC__)
  // Shared memory, don't need to allocate data for padding:
  if (block.padding() == 0) {
    auto exec_params       = default_shared_exec_params_t<Dims, T>{};
    auto [threads, blocks] = get_exec_size(block, exec_params);

    detail::invoke_static_shared_pipeline_multi<<<
      blocks, threads, 0, block.stream()
    >>>(
      block.begin(), 
      exec_params  ,
      pipeline     ,
      iter_or_ref(std::forward<Args>(args))...
    );
    ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
    return;
  }

  // Block has padding, need to dynamicall allocate:
  auto exec_params       = dynamic_device_params<Dims, T>(block.padding());
  auto [threads, blocks] = get_exec_size(block, exec_params);

  const auto mem_req = exec_params.template allocation_size<Dims>();

  detail::invoke_dynamic_shared_pipeline_multi<<<
    blocks, threads, mem_req, block.stream()
  >>>(
    block.begin(), 
    exec_params  ,
    pipeline     , 
    iter_or_ref(std::forward<Args>(args))...
  );
  ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
#endif // __CUDACC__
}

//==--- [non-variadic overload] --------------------------------------------==//

/// Invokes the \p pipeline on each element in the \p block. 
///
/// \param  block     The block to invoke the pipeline on.
/// \param  pipeline  The pipeline to invoke on the block.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Ops       The operations in the pipeline.
template <typename T, size_t Dims, typename... Ops>
auto invoke_pipeline(
  DeviceBlock<T, Dims>& block, const Pipeline<Ops...>& pipeline
) -> void {
#if defined(__CUDACC__)
  // Shared memory, don't need to allocate data for padding:
  if (block.padding() == 0) {
    auto exec_params       = default_shared_exec_params_t<Dims, T>{};
    auto [threads, blocks] = get_exec_size(block, exec_params);

    detail::invoke_static_shared_pipeline_multi<<<
      blocks, threads, 0, block.stream()
    >>>(
      block.begin(), exec_params, pipeline
    );
    ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
    return;
  }

  // Block has padding, need to dynamicall allocate:
  auto exec_params       = dynamic_device_params<Dims, T>(block.padding());
  auto [threads, blocks] = get_exec_size(block, exec_params);

  const auto mem_req = exec_params.template allocation_size<Dims>();

  detail::invoke_dynamic_shared_pipeline_multi<<<
    blocks, threads, mem_req, block.stream()
  >>>(
    block.begin(), exec_params, pipeline
  );
  ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
#endif // __CUDACC__
}

} // namespace ripple::kernel::detail

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_SHARED_CUDA__HPP

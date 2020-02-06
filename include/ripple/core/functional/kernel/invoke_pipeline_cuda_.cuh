//==--- ripple/core/functional/kernel/invoke_pipeline_cuda_.cuh - -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_pipeline_cuda_.cuh
/// \brief This file implements functionality to invoke a pipeline on various
///        container objects on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_CUDA__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_CUDA__HPP

#include "../pipeline.hpp"
#include <ripple/core/boundary/copy_loader.hpp>
#include <ripple/core/boundary/load_boundary.hpp>
#include <ripple/core/container/device_block.hpp>
#include <ripple/core/execution/execution_params.hpp>
#include <ripple/core/execution/execution_size.hpp>
#include <ripple/core/execution/execution_traits.hpp>
#include <ripple/core/execution/static_execution_params.hpp>
#include <ripple/core/execution/synchronize.hpp>
#include <ripple/core/execution/thread_index.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>

namespace ripple::kernel::cuda {
namespace detail {

//==--- [shared pipeline implementation] -----------------------------------==//

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
/// \param  exec_params    The parameters for the execution.
/// \param  pipeline       The pipeline to invoke on the iterator.
/// \tparam Dims           The number of dimensions in the block.
/// \tparam Iterator       The type of the iterator.
/// \tparam SharedIterator The type of the shared iterator.
/// \tparam ExeImpl        The implementation of the execution param interface.
/// \tparam Ops...         The type of the pipeline operations.
template <
  size_t      Dims          ,
  typename    Iterator      ,
  typename    SharedIterator,
  typename    ExecImpl      ,
  typename... Ops
>
ripple_device_only auto invoke_shared_pipeline(
  Iterator&&        it         ,
  SharedIterator&&  shared_it  ,
  ExecImpl&&        exec_params,
  Pipeline<Ops...>& pipeline
) -> void {
#if defined(__CUDACC__)
  bool in_range = true;
  // Shift higher dimensions ...
  unrolled_for<Dims>([&] (auto d) {
    constexpr auto dim     = d;
    const auto idx         = global_idx(dim);
    const auto must_shift = 
      idx             < it.size(dim)        &&
      thread_idx(dim) < shared_it.size(dim) &&
      in_range;
    if (must_shift) {
      it.shift(dim, idx);
      shared_it.shift(dim, thread_idx(dim) + shared_it.padding());
    } else {
      in_range = false;
    }
  });

  if (!in_range) {
    return;
  }

  using it_1_t = std::decay_t<decltype(*it)>;
  using it_2_t = std::decay_t<decltype(*shared_it)>;
  constexpr auto same_type =
    std::is_same_v<it_1_t, it_2_t> || std::is_convertible_v<it_1_t, it_2_t>;

  // Load in the padding data:
  if (same_type && shared_it.padding() > 0) {
    *shared_it = *it;
    load_internal_boundary<Dims>(it, shared_it);
  }
  sync_block();

  // Run each stage of the pipeline ...
  using pipeline_t = Pipeline<Ops...>;
  unrolled_for<pipeline_t::num_stages>([&] (auto i) {
    constexpr auto stage = size_t{i};
    pipeline.template get_stage<stage>().operator()(shared_it);
    sync_block();
  });
  *it = *shared_it;
#endif // __CUDACC__
}

/// Invokes the \p pipeline on a shared memory iterator whose data is set from
/// the global iterator \p it.
///
/// This function specifically creates the shared memory iterator, and then
/// calls the implementation which invokes each stage of the pipeline on the
/// iterator.
///
/// \param  it          The iterator to invoke the callable on.
/// \param  exec_params The parameters for the execution.
/// \param  pipeline    The pipeline to invoke on the iterator.
/// \tparam Iterator    The type of the global iterator.
/// \tparam ExeImpl     The implementation of the execution param interface.
/// \tparam Ops...      The type of the pipeline operations.
template <typename Iterator, typename ExecImpl, typename... Ops>
ripple_global auto invoke_static_shared_pipeline(
  Iterator it, ExecImpl exec_params, Pipeline<Ops...> pipeline
) -> void {
  constexpr auto dims       = it.dimensions();
  constexpr auto alloc_size = exec_params.template allocation_size<dims>();
  __shared__ char buffer[alloc_size];

  auto shared_it = 
    exec_params.template iterator<dims>(static_cast<void*>(buffer));

  invoke_shared_pipeline<dims>(it, shared_it, exec_params, pipeline);
}

} // namespace detail

//==--- [invoke pipeline] --------------------------------------------------==//

/// Invokes the \p pipeline on each element in the \p block.
///
/// \param  block     The block to invoke the pipeline on.
/// \param  pipeline  The pipeline to invoke on the block.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Ops       The operations in the pipeline.
template <typename T, size_t Dims, typename... Ops>
auto invoke(DeviceBlock<T, Dims>& block, Pipeline<Ops...>& pipeline) 
-> void {
#if defined(__CUDACC__)
  auto exec_params = default_shared_exec_params_t<Dims, T>{};
  auto [threads, blocks] = get_exec_size(block, exec_params);
  if (!block.uses_own_stream()) {
    detail::invoke_static_shared_pipeline<<<
      blocks, threads
    >>>(
      block.begin(), exec_params, pipeline
    );
    ripple_check_cuda_result(cudaDeviceSynchronize());
  } else {
    detail::invoke_static_shared_pipeline<<<
      blocks, threads, 0, block.stream()
    >>>(
      block.begin(), exec_params, pipeline
    );
    ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
  }
#endif // __CUDACC__
}

} // namespace ripple::kernel::cuda

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_CUDA__HPP


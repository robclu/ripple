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
#include <ripple/core/execution/detail/thread_index_impl_.hpp>
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
  unrolled_for<Dims>([&] (auto _dim) {
    constexpr auto dim = size_t{_dim};
    const     auto idx = global_idx(dim);

    // NOTE: No need for bounds check on shared iterator, since it's a 1-1
    // mapping to the global iterator, so we only need to test the global
    // iterator for the bound.
    if (in_range && idx < it.size(dim)/*&& thread_idx(dim) < threads*/) {
      it.shift(dim, idx);
      shared_it.shift(dim, thread_idx(dim) + shared_it.padding());
    } else {
      in_range = false;
    }
  });

  if (!in_range) { 
    return;
  }

  // Load in the internal data:
  using it_1_t = std::decay_t<decltype(*it)>;
  using it_2_t = std::decay_t<decltype(*shared_it)>;
  if (std::is_same_v<it_1_t, it_2_t> || std::is_convertible_v<it_1_t, it_2_t>) {
    *shared_it = *it;
  }

  // Load in the padding data:
  if (shared_it.padding() > 0) {
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

  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = size_t{d};
    ::ripple::detail::global_elements(dim) = it.size(dim);
  });

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
auto invoke_pipeline(
  DeviceBlock<T, Dims>& block, const Pipeline<Ops...>& pipeline
) -> void {
#if defined(__CUDACC__)
  auto exec_params       = default_shared_exec_params_t<Dims, T>{};
  auto [threads, blocks] = get_exec_size(block, exec_params);

  detail::invoke_static_shared_pipeline<<<blocks, threads, 0, block.stream()>>>(
    block.begin(), exec_params, pipeline
  );
  ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
#endif // __CUDACC__
}

//==--- [two blocks] -------------------------------------------------------==//

namespace detail {

/// Invokes each stage of the \p pipeline on the \p shared_it iterator, passing
/// the \p shared_it_other to each stage of the pipeline as an additional
/// argument.
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
/// \param  it                  The iterator to the global data.
/// \param  it_other            The other iterator to global data.
/// \param  shared_it           The iterator to the shared data.
/// \param  shared_it_other     The iterator to the other shared data.
/// \param  params              The parameters for the execution.
/// \param  oth_params          The other parameters for execution.
/// \param  pipeline            The pipeline to invoke on the iterator.
/// \tparam Dims                The number of dimensions in the block.
/// \tparam Iterator            The type of the iterator.
/// \tparam OtherIterator       The type of the other iterator.
/// \tparam SharedIterator      The type of the shared iterator.
/// \tparam SharedOtherIterator The type of the other shared iterator.
/// \tparam ExecImpl            The implementation type of the params.
/// \tparam OtherExecImpl       The implementation type of the other params.
/// \tparam Ops...              The type of the pipeline operations.
template <
  size_t      Dims               ,
  typename    Iterator           ,
  typename    OtherIterator      ,
  typename    SharedIterator     ,
  typename    SharedOtherIterator,
  typename    ExecImpl           ,
  typename    OtherExecImpl      ,
  typename... Ops
>
ripple_device_only auto invoke_shared_pipeline_multi(
  Iterator&&            it             ,
  OtherIterator&&       it_other       ,
  SharedIterator&&      shared_it      ,
  SharedOtherIterator&& shared_it_other,
  ExecImpl&&            params         ,
  OtherExecImpl&&       oth_params     ,
  Pipeline<Ops...>&     pipeline
) -> void {
#if defined(__CUDACC__)
  bool in_range = true;
  unrolled_for<Dims>([&] (auto _dim) {
    constexpr auto dim = size_t{_dim};
    const     auto idx = global_idx(dim);

    // NOTE: No need for bounds check on shared iterator, since it's a 1-1
    // mapping to the global iterator, so we only need to test the global
    // iterator for the bound.
    if (in_range && idx < it.size(dim)) {
      it.shift(dim, idx);
      it_other.shift(dim, idx);
      shared_it.shift(dim, thread_idx(dim) + shared_it.padding());
      shared_it_other.shift(dim, thread_idx(dim) + shared_it_other.padding());
    } else {
      in_range = false;
    }
  });

  if (!in_range) { 
    return;
  }

  // Load in the internal data:
  using it_1_t = std::decay_t<decltype(*it)>;
  using it_2_t = std::decay_t<decltype(*shared_it)>;
  if (std::is_same_v<it_1_t, it_2_t> || std::is_convertible_v<it_1_t, it_2_t>) {
    *shared_it = *it;
  }

  using ot_1_t = std::decay_t<decltype(*it_other)>;
  using ot_2_t = std::decay_t<decltype(*shared_it_other)>;
  if (std::is_same_v<ot_1_t, ot_2_t> || std::is_convertible_v<ot_1_t, ot_2_t>) {
    *shared_it_other = *it_other;
  }

  // Load in the padding data:
  if (shared_it.padding() > 0) {
    load_internal_boundary<Dims>(it, shared_it);
  }
  if (shared_it_other.padding() > 0) {
    load_internal_boundary<Dims>(it_other, shared_it_other);
  }
  sync_block();

  // Run each stage of the pipeline ...
  using pipeline_t = Pipeline<Ops...>;
  unrolled_for<pipeline_t::num_stages>([&] (auto i) {
    constexpr auto stage = size_t{i};
    pipeline.template get_stage<stage>().operator()(shared_it, shared_it_other);
    sync_block();
  });
  *it       = *shared_it;
  *it_other = *shared_it_other;
#endif // __CUDACC__
}

/// Invokes the \p pipeline on a shared memory iterator whose data is set from
/// the global iterator \p it. This also passes a shared memory iterator to the
/// other data.
///
/// This function specifically creates the shared memory iterator, and then
/// calls the implementation which invokes each stage of the pipeline on the
/// iterator.
///
/// \param  it              The iterator to invoke the callable on.
/// \param  it_other        The other iterator to pass to the callable.
/// \param  exec_params     The parameters for the execution.
/// \param  oth_exec_params The parameters for the execution for the other iter.
/// \param  pipeline        The pipeline to invoke on the iterator.
/// \tparam Iterator        The type of the global iterator.
/// \tparam OtherIterator   The type of the other iterator.
/// \tparam ExeImpl         The implementation of the execution param interface.
/// \tparam OthExeImpl      The implementation of the other exec params.
/// \tparam Ops...          The type of the pipeline operations.
template <
  typename    Iterator     ,
  typename    OtherIterator,
  typename    ExecImpl     ,
  typename    OtherExecImpl,
  typename... Ops
>
ripple_global auto invoke_static_shared_pipeline_two(
  Iterator         it        , 
  OtherIterator    it_other  ,
  ExecImpl         params    ,
  OtherExecImpl    oth_params,
  Pipeline<Ops...> pipeline
) -> void {
  constexpr auto dims         = it.dimensions();
  constexpr auto alloc_size_a = params.template allocation_size<dims>();
  constexpr auto alloc_size_b = oth_params.template allocation_size<dims>();
  __shared__ char buffer[alloc_size_a + alloc_size_b];

  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = size_t{d};
    ::ripple::detail::global_elements(dim) = it.size(dim);
  });

  auto shared_it       = params.template iterator<dims>(
    static_cast<void*>(buffer)
  );
  auto shared_it_other = oth_params.template iterator<dims>(
    static_cast<void*>(buffer + alloc_size_a)
  );

  invoke_shared_pipeline_multi<dims>(
    it, it_other, shared_it, shared_it_other, params, oth_params, pipeline
  );
}

} // namespace detail

/// Invokes the \p pipeline on each element in the \p block, passing an iterator
/// to the elements in the \p other block.
///
/// \param  block     The block to invoke the pipeline on.
/// \pram   other     The other block to pass to the pipeline.
/// \param  pipeline  The pipeline to invoke on the block.
/// \tparam T         The type of the data in the block.
/// \param  U         The type of the data in the other block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Ops       The operations in the pipeline.
template <typename T, typename U, size_t Dims, typename... Ops>
auto invoke_pipeline(
  DeviceBlock<T, Dims>&   block, 
  DeviceBlock<U, Dims>&   other,
  const Pipeline<Ops...>& pipeline
)  -> void {
#if defined(__CUDACC__)
  using t_contig_t = as_contiguous_view_t<T>;
  using u_contig_t = as_contiguous_view_t<U>;

  auto exec_params       = default_shared_exec_params_t<Dims, t_contig_t>{};
  auto oth_exec_params   = default_shared_exec_params_t<Dims, u_contig_t>{};
  auto [threads, blocks] = get_exec_size(block, exec_params);

  detail::invoke_static_shared_pipeline_two<<<
      blocks, threads, 0, block.stream()
  >>>(
    block.begin(), other.begin(), exec_params, oth_exec_params, pipeline
  );
  ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
#endif // __CUDACC__
}

//==--- [three blocks] -----------------------------------------------------==//

namespace detail {

/// Invokes each stage of the \p pipeline on the \p shared_it iterator, passing
/// the \p shared_it_other to each stage of the pipeline as an additional
/// argument.
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
/// \param  it                  The iterator to the global data.
/// \param  it_other            The other iterator to global data.
/// \param  shared_it           The iterator to the shared data.
/// \param  shared_it_other     The iterator to the other shared data.
/// \param  params              The parameters for the execution.
/// \param  oth_params          The other parameters for execution.
/// \param  pipeline            The pipeline to invoke on the iterator.
/// \tparam Dims                The number of dimensions in the block.
/// \tparam Iterator            The type of the iterator.
/// \tparam OtherIterator       The type of the other iterator.
/// \tparam SharedIterator      The type of the shared iterator.
/// \tparam SharedOtherIterator The type of the other shared iterator.
/// \tparam ExecImpl            The implementation type of the params.
/// \tparam OtherExecImpl       The implementation type of the other params.
/// \tparam Ops...              The type of the pipeline operations.
template <
  size_t      Dims               ,
  typename    Iterator           ,
  typename    OtherIterator      ,
  typename    SharedIterator     ,
  typename    SharedOtherIterator,
  typename    LastIterator       ,
  typename    ExecImpl           ,
  typename    OtherExecImpl      ,
  typename... Ops                ,
  typename... Args
>
ripple_device_only auto invoke_shared_pipeline_three(
  Iterator&&            it             ,
  OtherIterator&&       it_other       ,
  SharedIterator&&      shared_it      ,
  SharedOtherIterator&& shared_it_other,
  LastIterator&&        it_last        ,
  ExecImpl&&            params         ,
  OtherExecImpl&&       oth_params     ,
  Pipeline<Ops...>&     pipeline       ,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  bool in_range = true;
  unrolled_for<Dims>([&] (auto _dim) {
    constexpr auto dim = size_t{_dim};
    const     auto idx = global_idx(dim);

    // NOTE: No need for bounds check on shared iterator, since it's a 1-1
    // mapping to the global iterator, so we only need to test the global
    // iterator for the bound.
    if (in_range && idx < it.size(dim)) {
      it.shift(dim, idx);
      it_other.shift(dim, idx);
      it_last.shift(dim, idx);
      shared_it.shift(dim, thread_idx(dim) + shared_it.padding());
      shared_it_other.shift(dim, thread_idx(dim) + shared_it_other.padding());
    } else {
      in_range = false;
    }
  });

  if (!in_range) { 
    return;
  }

  // Load in the internal data:
  using it_1_t = std::decay_t<decltype(*it)>;
  using it_2_t = std::decay_t<decltype(*shared_it)>;
  if (std::is_same_v<it_1_t, it_2_t> || std::is_convertible_v<it_1_t, it_2_t>) {
    *shared_it = *it;
  }

  using ot_1_t = std::decay_t<decltype(*it_other)>;
  using ot_2_t = std::decay_t<decltype(*shared_it_other)>;
  if (std::is_same_v<ot_1_t, ot_2_t> || std::is_convertible_v<ot_1_t, ot_2_t>) {
    *shared_it_other = *it_other;
  }

  // Load in the padding data:
  if (shared_it.padding() > 0) {
    load_internal_boundary<Dims>(it, shared_it);
  }
  if (shared_it_other.padding() > 0) {
    load_internal_boundary<Dims>(it_other, shared_it_other);
  }
  sync_block();

  // Run each stage of the pipeline ...
  using pipeline_t = Pipeline<Ops...>;
  unrolled_for<pipeline_t::num_stages>([&] (auto i) {
    constexpr auto stage = size_t{i};
    pipeline.template get_stage<stage>().operator()(
      shared_it, shared_it_other, it_last, std::forward<Args>(args)...
    );
    sync_block();
  });
  *it       = *shared_it;
  *it_other = *shared_it_other;
#endif // __CUDACC__
}

/// Invokes the \p pipeline on a shared memory iterator whose data is set from
/// the global iterator \p it. This also passes a shared memory iterator to the
/// other data.
///
/// This function specifically creates the shared memory iterator, and then
/// calls the implementation which invokes each stage of the pipeline on the
/// iterator.
///
/// \param  it              The iterator to invoke the callable on.
/// \param  it_other        The other iterator to pass to the callable.
/// \param  exec_params     The parameters for the execution.
/// \param  oth_exec_params The parameters for the execution for the other iter.
/// \param  pipeline        The pipeline to invoke on the iterator.
/// \tparam Iterator        The type of the global iterator.
/// \tparam OtherIterator   The type of the other iterator.
/// \tparam ExeImpl         The implementation of the execution param interface.
/// \tparam OthExeImpl      The implementation of the other exec params.
/// \tparam Ops...          The type of the pipeline operations.
template <
  typename    Iterator     ,
  typename    OtherIterator,
  typename    LastIterator ,
  typename    ExecImpl     ,
  typename    OtherExecImpl,
  typename... Ops          ,
  typename... Args
>
ripple_global auto invoke_static_shared_pipeline_three(
  Iterator         it        , 
  OtherIterator    it_other  ,
  LastIterator     it_last   ,
  ExecImpl         params    ,
  OtherExecImpl    oth_params,
  Pipeline<Ops...> pipeline  ,
  Args...          args
) -> void {
  constexpr auto dims         = it.dimensions();
  constexpr auto alloc_size_a = params.template allocation_size<dims>();
  constexpr auto alloc_size_b = oth_params.template allocation_size<dims>();
  __shared__ char buffer[alloc_size_a + alloc_size_b];

  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = size_t{d};
    ::ripple::detail::global_elements(dim) = it.size(dim);
  });

  auto shared_it       = params.template iterator<dims>(
    static_cast<void*>(buffer)
  );
  auto shared_it_other = oth_params.template iterator<dims>(
    static_cast<void*>(buffer + alloc_size_a)
  );

  invoke_shared_pipeline_three<dims>(
    it             , 
    it_other       ,
    shared_it      ,
    shared_it_other,
    it_last        ,
    params         ,
    oth_params     ,
    pipeline       ,
    args...
  );
}

} // namespace detail

/// Invokes the \p pipeline on each element in the \p block, passing an iterator
/// to the elements in the \p other block.
///
/// \param  block    The block to invoke the pipeline on.
/// \param  other    The other block to pass to the pipeline.
/// \param  last     The last block to pass to the pipeline.
/// \param  pipeline The pipeline to invoke on the block.
/// \param  args     Additional arguments to provide.
/// \tparam T        The data type for the block.
/// \tparam U        The data type for the other block.
/// \tparam V        The data type for the last block.
/// \tparam Dims     The number of dimensions in the block.
/// \tparam Ops      The type of the pipeline operations.
/// \tparam Args     The type of additional arguments./
template <
  typename    T   ,
  typename    U   ,
  typename    V   ,
  size_t      Dims,
  typename... Ops ,
  typename... Args
>
auto invoke_pipeline(
  DeviceBlock<T, Dims>&   block   , 
  DeviceBlock<U, Dims>&   other   ,
  DeviceBlock<V, Dims>&   last    ,
  const Pipeline<Ops...>& pipeline,
  Args&&...               args
)  -> void {
#if defined(__CUDACC__)
  using t_contig_t = as_contiguous_view_t<T>;
  using u_contig_t = as_contiguous_view_t<U>;

  auto exec_params       = default_shared_exec_params_t<Dims, t_contig_t>{};
  auto oth_exec_params   = default_shared_exec_params_t<Dims, u_contig_t>{};
  auto [threads, blocks] = get_exec_size(block, exec_params);

  detail::invoke_static_shared_pipeline_three<<<
      blocks, threads, 0, block.stream()
  >>>(
    block.begin()  ,
    other.begin()  ,
    last.begin()   ,
    exec_params    ,
    oth_exec_params,
    pipeline       ,
    args...
  );
  ripple_check_cuda_result(cudaStreamSynchronize(block.stream()));
#endif // __CUDACC__
}


} // namespace ripple::kernel::cuda

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_CUDA__HPP


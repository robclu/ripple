//==--- ripple/functional/kernel/invoke_cuda_.cuh ---------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_cuda_.cuh
/// \brief This file implements functionality to invoke a callable object on
///        a cuda device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CUDA__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CUDA__HPP

#include <ripple/boundary/copy_loader.hpp>
#include <ripple/boundary/load_boundary.hpp>
#include <ripple/container/device_block.hpp>
#include <ripple/execution/execution_params.hpp>
#include <ripple/execution/execution_size.hpp>
#include <ripple/execution/execution_traits.hpp>
#include <ripple/execution/static_execution_params.hpp>
#include <ripple/execution/synchronize.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/iterator/iterator_traits.hpp>
#include <ripple/perf/perf_traits.hpp>

namespace ripple::kernel::cuda {
namespace detail {

//==--- [non shared memory] ------------------------------------------------==//

/// Invokes the \p callale on the iterator, shifting the iterator by the thread
/// index.
///
/// \param  it          The iterator to invoke the callable on.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    Iterator,
  typename    Callable,
  typename... Args    ,
  non_iterator_enable_t<Callable> = 0
>
ripple_global auto invoke(Iterator it, Callable callable, Args... args)
-> void {
  constexpr auto dims = it.dimensions();
  bool in_range       = true;

  // Shift higher dimensions ...
  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = d;
    const auto     idx = global_idx(dim);
    if (idx < it.size(dim) && in_range) {
      it.shift(dim, idx);
    } else {
      in_range = false;
    }
  });

  if (!in_range) {
    return;
  }

  callable(it, args...);
}

/// Invokes the \p callale on the iterator \p it_1 and passing the iterator 
/// \p it_2 as the second argument, shifting both of the iterators by the thread
/// indices in each of the dimensions which are valid for the iterator.
///
/// \param  it_1        The iterator to invoke the callable on.
/// \param  it_2        An additioanl iterator to pass to the callable.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam Iterator1   The type of the first iterator.
/// \tparam Iterator2   The type of the second iterator.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    Iterator1, 
  typename    Iterator2,
  typename    Callable ,
  typename... Args     ,
  iterator_enable_t<Iterator2> = 0
>
ripple_global auto invoke(
  Iterator1 it_1    ,
  Iterator2 it_2    ,
  Callable  callable, 
  Args...   args
) -> void {
  constexpr auto dims_1 = it_1.dimensions();
  constexpr auto dims_2 = it_2.dimensions();
  bool in_range         = true;

  // Shift higher dimensions ...
  unrolled_for<dims_1>([&] (auto d) {
    constexpr auto dim = d;
    const auto     idx = global_idx(dim);
    if (idx < it_1.size(dim) && in_range) {
      it_1.shift(dim, idx);
    } else {
      in_range = false;
    }

    if constexpr (dim < dims_2) {
      it_2.shift(dim, idx);
    }
  });

  if (!in_range) {
    return;
  }

  callable(it_1, it_2, args...);
}

//--- [shared memory] --------------------------------------------------------//

/// Invokes the \p callable on the iterator, shifting the iterator by the thread
/// index in the dim_x (0) dimension, and passing the \p shared_it iterator.
///
/// \param  it          The iterator to invoke the callable on.
/// \param  exec_params The parameters for the execution.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExeImpl     The implementation of the execution param interface.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  std::size_t Dims          ,
  typename    Iterator      ,
  typename    SharedIterator,
  typename    ExecImpl      ,
  typename    Callable      ,
  typename... Args
>
ripple_device_only auto invoke_shared(
  Iterator&&       it         ,
  SharedIterator&& shared_it  ,
  ExecImpl&&       exec_params,
  Callable&&       callable   ,
  Args&&...        args  
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
    load_internal_boundary<Dims>(it, shared_it);
  }
  sync_block();

  callable(it, shared_it, args...);
#endif // __CUDACC__
}

/// Invokes the \p callale on the iterator, shifting the iterator by the thread
/// index.
///
/// \param  it          The iterator to invoke the callable on.
/// \param  exec_params The parameters for the execution.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExeImpl     The implementation of the execution param interface.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    Iterator,
  typename    ExecImpl,
  typename    Callable,
  typename... Args
>
ripple_global auto invoke_static_shared(
  Iterator  it         ,
  ExecImpl  exec_params,
  Callable  callable   ,
  Args...   args
) -> void {
  constexpr auto dims       = it.dimensions();
  constexpr auto alloc_size = exec_params.template allocation_size<dims>();
  __shared__ char buffer[alloc_size];

  auto shared_it = 
    exec_params.template iterator<dims>(static_cast<void*>(buffer));

  invoke_shared<dims>(it, shared_it, exec_params, callable, args...);
}

/// Invokes the \p callale on the iterator, shifting the iterator by the thread
/// index. It also creates dynamic shared memory, and then creates an iterator
/// over the shared memory space.
///
/// \param  it          The iterator to invoke the callable on.
/// \param  exec_params The parameters for the execution.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam Iteartor    The type of the iterator.
/// \tparam ExecImpl    The implementation of the execution param interface.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    Iterator,
  typename    ExecImpl,
  typename    Callable,
  typename... Args
>
ripple_global auto invoke_dynamic_shared(
  Iterator  it         ,
  ExecImpl  exec_params,
  Callable  callable   ,
  Args...   args
) -> void {
  constexpr auto dims = it.dimensions();
  extern __shared__ char buffer[];
  auto shared_it = 
    exec_params.template iterator<dims>(static_cast<void*>(buffer));

  invoke_shared<dims>(it, shared_it, exec_params, callable, args...);
}

} // namespace detail

//==--- [simple invoke] ----------------------------------------------------==//

/// Invokes the \p callale on each element in the \p block.
///
/// \param  block     The block to invoke the callable on.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    Callable,
  typename... Args    ,
  non_exec_param_enable_t<Callable> = 0
>
auto invoke(DeviceBlock<T, Dims>& block, Callable&& callable, Args&&... args)
-> void {
#if defined(__CUDACC__)
  using exec_params_t = default_exec_params_t<Dims>;
  auto [threads, blocks] = get_exec_size(block, exec_params_t{});
  detail::invoke<<<blocks, threads>>>(
    block.begin(), callable, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

/// Invokes the \p callale on each element in the \p block_1.
///
/// \param  block_1   The first block to invoke the callable on.
/// \param  block_2   The second block to pass to the callable.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T1        The type of the data in the first block.
/// \tparam T2        The type of the data in the second block.
/// \tparam Dims1     The number of dimensions in the first block.
/// \tparam Dims2     The number of dimensions in the second block.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    T1      ,
  std::size_t Dims1   ,
  typename    T2      ,
  std::size_t Dims2   ,
  typename    Callable,
  typename... Args    ,
  non_exec_param_enable_t<Callable> = 0
>
auto invoke(
  DeviceBlock<T1, Dims1>& block_1 , 
  DeviceBlock<T2, Dims2>& block_2 , 
  Callable&&              callable,
  Args&&...               args
) -> void {
#if defined(__CUDACC__)
  using exec_params_t = default_exec_params_t<Dims1>;
  auto [threads, blocks] = get_exec_size(block_1, exec_params_t{});
  detail::invoke<<<blocks, threads>>>(
    block_1.begin(), block_2.begin(), callable, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

//==--- [exec invoke] ------------------------------------------------------==//

/// Invokes the \p callale on each element in the \p block, using the execution
/// parameters \p params.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params Parameters for the execution.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExecImpl    The type of the execution param implementation.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExecImpl,
  typename    Callable,
  typename... Args    ,
  non_shared_enable_t<ExecImpl> = 0
>
auto invoke(
  DeviceBlock<T, Dims>& block      ,
  ExecImpl&&            exec_params,
  Callable&&            callable   ,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  auto [threads, blocks] = get_exec_size(block, exec_params);
  detail::invoke<<<blocks, threads>>>(
    block.begin(), callable, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

//--- [static shared] ------------------------------------------------------==//

/// Invokes the \p callale on each element in the \p block, using the execution
/// parameters to define the execution space.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params Parameters for the execution.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExecImpl    The type of the execution param implementation.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExeImpl ,
  typename    Callable,
  typename... Args    ,
  static_shared_enable_t<ExeImpl> = 0
>
auto invoke(
  DeviceBlock<T, Dims>& block      ,
  ExeImpl&&             exec_params,
  Callable&&            callable   ,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  auto [threads, blocks] = get_exec_size(block, exec_params);
  detail::invoke_static_shared<<<blocks, threads>>>(
    block.begin(), exec_params, callable, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

//--- [dynamic shared] -----------------------------------------------------==//

/// Invokes the \p callale on each element in the \p block, using the provided
/// execution space.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params Parameters for the execution.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExecImpl    The type of the execution param implementation.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExeImpl ,
  typename    Callable,
  typename... Args    ,
  dynamic_shared_enable_t<ExeImpl> = 0
>
auto invoke(
  DeviceBlock<T, Dims>& block      ,
  ExeImpl&&             exec_params,
  Callable&&            callable   ,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  auto [threads, blocks] = get_exec_size(block, exec_params);
  auto alloc_size = exec_params.template allocation_size<Dims>();
  detail::invoke_dynamic_shared<<<blocks, threads, alloc_size>>>(
    block.begin(), exec_params, callable, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif
}

namespace bench {

/// Invokes the \p callable on each element in the \p block, profiling the
/// kernel filling the \p event with profiling information.
///
/// \param  block     The block to invoke the callable on.
/// \param  event     The event to fill with the profiling information.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    Callable,
  typename... Args    ,
  non_exec_param_enable_t<Callable> = 0
>
auto invoke(
  DeviceBlock<T, Dims>& block   ,
  Event&                event   ,
  Callable&&            callable,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  using exec_params_t = default_exec_params_t<Dims>;
  exec_params_t exec_params;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto [threads, blocks] = get_exec_size(block, exec_params);

  cudaEventRecord(start);
  detail::invoke<<<blocks, threads>>>(
    block.begin(), callable, args...
  );
  cudaEventRecord(stop);

  ripple_check_cuda_result(cudaDeviceSynchronize());
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&event.elapsed_time_ms, start, stop);
#endif // __CUDACC__
}

//==--- [nonshared] --------------------------------------------------------==//

/// Invokes the \p callable on each element in the \p block, profiling the
/// kernel filling the \p event with profiling information.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params Parameters for the execution.
/// \param  event       The event to fill with the profiling information.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExeImpl     The type of the execution param implementation.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExeImpl ,
  typename    Callable,
  typename... Args    ,
  non_shared_enable_t<ExeImpl> = 0
>
auto invoke(
  DeviceBlock<T, Dims>&      block      ,
  Event&                     event      ,
  ExeImpl&&                  exec_params,
  Callable&&                 callable   ,
  Args&&...                  args
) -> void {
#if defined(__CUDACC__)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto [threads, blocks] = get_exec_size(block, exec_params);

  cudaEventRecord(start);
  detail::invoke<<<blocks, threads>>>(
    block.begin(), callable, args...
  );
  cudaEventRecord(stop);

  ripple_check_cuda_result(cudaDeviceSynchronize());
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&event.elapsed_time_ms, start, stop);
#endif // __CUDACC__
}

//==--- [static shared] ----------------------------------------------------==//

/// Invokes the \p callale on each element in the \p block, profiling the
/// kernel filling the \p event with profiling information.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params Parameters for the execution.
/// \param  event       The event to fill with the profiling information.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExeImpl     The type of the execution param implementation.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExeImpl ,
  typename    Callable,
  typename... Args    ,
  static_shared_enable_t<ExeImpl> = 0
>
auto invoke(
  DeviceBlock<T, Dims>& block      ,
  Event&                event      ,
  ExeImpl&&             exec_params,
  Callable&&            callable   ,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto [threads, blocks] = get_exec_size(block, exec_params);

  cudaEventRecord(start);
  detail::invoke_static_shared<<<blocks, threads>>>(
    block.begin(), exec_params, callable, args...
  );
  cudaEventRecord(stop);

  ripple_check_cuda_result(cudaDeviceSynchronize());
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&event.elapsed_time_ms, start, stop);
#endif // __CUDACC__
}

//--- [dynamic shared] -----------------------------------------------------==//

/// Invokes the \p callale on each element in the \p block, profiling the
/// kernel filling the \p event with profiling information.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params Parameters for the execution.
/// \param  event       The event to fill with the profiling information.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExeImpl     The type of the execution param implementation.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExeImpl ,
  typename    Callable,
  typename... Args    ,
  dynamic_shared_enable_t<ExeImpl> = 0
>
auto invoke(
  DeviceBlock<T, Dims>& block      ,
  Event&                event      ,
  ExeImpl&&             exec_params,
  Callable&&            callable   ,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto [threads, blocks] = get_exec_size(block, exec_params);
  auto alloc_size = exec_params.template allocation_size<Dims>();

  cudaEventRecord(start);
  detail::invoke_dynamic_shared<<<blocks, threads, alloc_size>>>(
    block.begin(), exec_params, callable, args...
  );
  cudaEventRecord(stop);

  ripple_check_cuda_result(cudaDeviceSynchronize());
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&event.elapsed_time_ms, start, stop);
#endif
}

} // namespace bench

} // namespace ripple::kernel::cuda

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CUDA__HPP


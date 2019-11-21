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

#include <ripple/container/device_block.hpp>
#include <ripple/execution/execution_params.hpp>
#include <ripple/execution/execution_size.hpp>
#include <ripple/execution/execution_traits.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/perf/perf_traits.hpp>

namespace ripple::kernel::cuda {
namespace detail {

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
template <typename Iterator, typename Callable, typename... Args>
ripple_global auto invoke(Iterator it, Callable callable, Args... args)
-> void {
  constexpr auto dims = it.dimensions();
  bool in_range       = true;

  // Shift higher dimensions ...
  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = d;
    const auto     idx = global_idx(dim);
    if (idx < it.size(dim) && in_range) {
      it.shift(dim, idx + it.padding());
    } else {
      in_range = false;
    }
  });

  if (!in_range) {
    return;
  }

  // Offset in the last dimension, and invoke callable:
  callable(it, args...);
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
  bool in_range = true;
  // Shift higher dimensions ...
  unrolled_for<Dims>([&] (auto d) {
    constexpr auto dim = Dims - 1 - d;
    const auto     idx = global_idx(dim);
    if (idx < it.size(dim) && in_range) {
      it.shift(dim, idx + it.padding());
      shared_it.shift(dim, thread_idx(dim) + shared_it.padding());
    } else {
      in_range = false;
    }
  });

  if (!in_range)
    return;

  callable(it, shared_it, args...);
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
  auto shared_it = exec_params.iterator(static_cast<void*>(buffer));

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
  auto shared_it = exec_params.iterator(static_cast<void*>(buffer));

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


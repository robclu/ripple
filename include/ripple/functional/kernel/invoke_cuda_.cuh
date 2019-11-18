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
#include <ripple/multidim/thread_index.hpp>
#include <ripple/time/event.hpp>

namespace ripple::kernel::cuda {
namespace detail {

/// Invokes the \p callable on the iterator, shifting the iterator by the thread
/// index.
///
/// This overload is enables when the execution parameters do not specify any
/// grain size.
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
  typename... Args    ,
  single_grain_enable_t<ExecImpl> = 0
>
ripple_device_only auto invoke_with_grain(
  Iterator&&      it         ,
  const ExecImpl& exec_params,
  Callable&&      callable   ,
  Args&&...       args     
) -> void {
  if (grid_idx(dim_x) >= it.size(dim_x)) {
    return;
  }
  callable(it.offset(dim_x, grid_idx(dim_x)), args...); 
}

/// Invokes the \p callable on the iterator, shifting the iterator by the thread
/// index.
///
/// This overload is enables when the execution parameters specify a grain size,
/// in which case the callable is invoked `exec_params.grain_size()` times.
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
  typename... Args    ,
  multi_grain_enable_t<ExecImpl> = 0
>
ripple_device_only auto invoke_with_grain(
  Iterator&& it         ,
  ExecImpl&  exec_params,
  Callable&& callable,
  Args&&...  args
) -> void {
  std::size_t idx = 0;
  constexpr auto grain_size = exec_params.grain_size();
  for (auto grain_idx : range(grain_size)) {
    exec_params.grain_index() = grain_idx;
    idx                       = grid_idx(dim_x, exec_params);
    if (idx >= it.size(dim_x)) {
      return;
    }

    callable(it.offset(dim_x, idx), exec_params, args...);
  }
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
ripple_global auto invoke(
  Iterator  it,
  ExecImpl  exec_params,
  Callable  callable,
  Args...   args
) -> void {
  constexpr auto dims = it.dimensions();
  bool in_range       = true;

  // Shift higher dimensions ...
  unrolled_for<dims - 1>([&] (auto d) {
    constexpr auto dim = d + 1;
    const auto     idx = grid_idx(dim);
    if (idx < it.size(dim) && in_range) {
      it.shift(dim, idx);
    } else {
      in_range = false;
    }
  });

  if (!in_range)
    return;

  // Invoke the callable on the last dimension ...
  invoke_with_grain(
    it                              ,
    exec_params                     ,
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
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
template <typename T, std::size_t Dims, typename Callable, typename... Args>
auto invoke(DeviceBlock<T, Dims>& block, Callable&& callable, Args&&... args)
-> void {
#if defined(__CUDACC__)
  using exec_params_t = default_exec_params_t<Dims>;
  auto [threads, blocks] = get_exec_size(block, exec_params_t{});
  detail::invoke<<<blocks, threads>>>(
    block.begin(), exec_params_t{}, callable, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

//==--- [event invoke] -----------------------------------------------------==//

/// Invokes the \p callale on each element in the \p block, profiling the
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
template <typename T, std::size_t Dims, typename Callable, typename... Args>
auto invoke(
  DeviceBlock<T, Dims>& block,
  Event&                event,
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
    block.begin(), exec_params, callable, args...
  );
  cudaEventRecord(stop);

  ripple_check_cuda_result(cudaDeviceSynchronize());
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&event.elapsed_time_ms, start, stop);
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
/// \tparam ExeImpl     The type of the execution param implementation.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T,
  std::size_t Dims,
  typename    ExeImpl,
  typename    Callable,
  typename... Args
>
auto invoke(
  DeviceBlock<T, Dims>&      block,
  const ExecParams<ExeImpl>& exec_params,
  Callable&&                 callable,
  Args&&...                  args
) -> void {
#if defined(__CUDACC__)
  auto [threads, blocks] = get_exec_size(block, exec_params);
  detail::invoke<<<blocks, threads>>>(
    block.begin(), exec_params, callable, args...
  );
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

} // namespace ripple::kernel::cuda

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CUDA__HPP


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
#include <ripple/execution/execution_size.hpp>
#include <ripple/execution/execution_traits.hpp>
#include <ripple/multidim/thread_index.hpp>
#include <ripple/time/event.hpp>

namespace ripple::kernel::cuda {
namespace detail {

/// Invokes the \p callale on the iterator, shifting the iterator by the thread
/// index.
///
/// \param  it        The iterator to invoke the callable on.
/// \param  params    The parameters for the execution.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Params    The type of the execution paramters.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    Iterator,
  typename    Params,
  typename    Callable,
  typename... Args
>
ripple_global auto invoke(
  Iterator it,
  Params   params,
  Callable callable,
  Args...  args
) -> void {
  using params_t      = std::decay_t<Params>;
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

  if constexpr (params_t::grain <= 1) {
    if (grid_idx(dim_x) >= it.size(dim_x)) {
      return;
    }
    callable(it.offset(dim_x, grid_idx(dim_x)), args...);
  } else {
    std::size_t idx = 0;
    for (auto grain_idx : range(params_t::grain)) {
      params.grain_index = grain_idx;
      idx                = grid_idx(dim_x, params);
      if (idx >= it.size(dim_x)) {
        return;
      }

      callable(it.offset(dim_x, idx), params, args...);
    }
  }
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
  using params_t = default_exec_params_t<Dims>;
  auto [threads, blocks] = get_exec_size(block, params_t{});
  detail::invoke<<<blocks, threads>>>(block.begin(), params_t{}, callable, args...);
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
  using params_t = default_exec_params_t<Dims>;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto [threads, blocks] = get_exec_size(block, params_t{});

  cudaEventRecord(start);
  detail::invoke<<<blocks, threads>>>(block.begin(), params_t{}, callable, args...);
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
/// \param  block     The block to invoke the callable on.
/// \param  params    The parameters for the execution.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Params    The type of the execution paramters.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    T,
  std::size_t Dims,
  typename    Params,
  typename    Callable,
  typename... Args
>
auto invoke(
  DeviceBlock<T, Dims>& block,
  Params&&              params,
  Callable&&            callable,
  Args&&...             args
) -> void {
#if defined(__CUDACC__)
  auto [threads, blocks] = get_exec_size(block, params);
  detail::invoke<<<blocks, threads>>>(block.begin(), params, callable, args...);
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

} // namespace ripple::kernel::cuda


#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CUDA__HPP


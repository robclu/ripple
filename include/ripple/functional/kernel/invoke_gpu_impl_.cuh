/**=--- ripple/kernel/detail/invoke_gpu_impl_.cuh ---------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  invoke_gpu_impl_.cuh
 * \brief This file implements functionality to invoke a callable object on
 *        blocks on the gpu.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GPU_IMPL__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GPU_IMPL__HPP

#include <ripple/container/device_block.hpp>
#include <ripple/execution/execution_params.hpp>
#include <ripple/execution/execution_size.hpp>
#include <ripple/execution/execution_traits.hpp>
#include <ripple/execution/static_execution_params.hpp>
#include <ripple/execution/synchronize.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/iterator/iterator_traits.hpp>
#include <ripple/padding/copy_loader.hpp>
#include <ripple/padding/load_padding.hpp>

namespace ripple::kernel::gpu {
namespace detail {

/*==--- [non shared memory] ------------------------------------------------==*/

// clang-format off
/**
 * Invokes the given callable on the iterator, shifting the iterator by the 
 * thread index.
 *
 * \param  it        The iterator to invoke the callable on.
 * \param  callable  The callable object.
 * \param  args      Arguments for the callable.
 * \tparam Iterator  The type of the iterator.
 * \tparam Callable  The callable object to invoke.
 * \tparam Args      The type of the arguments for the invocation.
 */
template <
  typename    Iterator,
  typename    Callable,
  non_iterator_enable_t<Callable> = 0,
  typename... Args>
ripple_global auto
invoke(Iterator it, Callable callable, Args... args) -> void {
  // clang-format on
  constexpr auto dims     = it.dimensions();
  bool           in_range = true;

  // Shift higher dimensions ...
  unrolled_for<dims>([&](auto d) {
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

} // namespace detail

/*==--- [invoke interface] -------------------------------------------------==*/

// clang-format off
/**
 * Invokes the given callale on each element in the given block.
 *
 * \param  block     The block to invoke the callable on.
 * \param  callable  The callable object.
 * \param  args      Arguments for the callable.
 * \tparam T         The type of the data in the block.
 * \tparam Dims      The number of dimensions in the block.
 * \tparam Callable  The callable object to invoke.
 * \tparam Args      The type of the arguments for the invocation.
 */
template <
  typename    T,
  size_t      Dims,
  typename    Callable,
  typename... Args,
  non_exec_param_enable_t<Callable> = 0>
auto invoke(DeviceBlock<T, Dims>& block, Callable&& callable, Args&&... args)
  noexcept -> void {
  // clang-format on
  using ExecParams       = default_exec_params_t<Dims>;
  auto [threads, blocks] = get_exec_size(block, ExecParams{});

  ripple_if_cuda(detail::invoke<<<blocks, threads, 0, block.stream()>>>(
    block.begin(), callable, args...));
}

} // namespace ripple::kernel::gpu

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GPU_IMPL__HPP

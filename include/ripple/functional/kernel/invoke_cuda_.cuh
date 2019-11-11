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

#include <ripple/execution/execution_size.hpp>
#include <ripple/multidim/thread_index.hpp>
#include <ripple/container/device_block.hpp>

namespace ripple::kernel::cuda {
namespace detail {

/// Invokes the \p callale on the iterator, shifting the iterator by the thread
/// index.
///
/// \param  it        The iterator to invoke the callable on.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <typename Iterator, typename Callable, typename... Args>
ripple_global auto invoke(Iterator it, Callable callable, Args... args)
-> void {
  constexpr auto dims = it.dimensions();
  bool in_range = true;
  unrolled_for<dims>([&] (auto dim) {
    if (flattened_idx(dim) < it.size(dim) && in_range) {
      it.shift(dim, flattened_idx(dim));
    } else {
      in_range = false;
    }
  });
  if (in_range) {
    callable(it, args...);
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
  auto [threads, blocks] = exec::get_exec_size(block);
  detail::invoke<<<blocks, threads>>>(block.begin(), callable, args...);
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

} // namespace ripple::kernel::cuda


#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CUDA__HPP


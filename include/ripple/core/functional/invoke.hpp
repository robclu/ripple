//==--- ripple/core/functional/invoke.hpp ------------------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke.hpp
/// \brief This file implements functionality to invoke a functor on a block.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_INVOKE_HPP
#define RIPPLE_FUNCTIONAL_INVOKE_HPP

#include "kernel/invoke_cpp_.hpp"
#include "kernel/invoke_block_impl_.cuh"
#include "kernel/invoke_generic_impl_.cuh"
#include "kernel/invoke_generic_impl_.hpp"
#include <ripple/core/execution/execution_traits.hpp>

namespace ripple {

/*==--- [generic invoke] ---------------------------------------------------==*/

/**
 * Perfoms an invocation of the invocable, forwarding and possibly manipulating
 * the passed arguments.
 *
 * This will look at the types of the arguments, and for any which are Block
 * types, or BlockEnabled, will pass them as offset iterators to the invocable.
 *
 * If any of the arguments are wrapped with shared wrapper types, they are
 * passed as iterators over shared memory.
 *
 * \note This overload is for specifying the execution kind at compile time.
 *
 * \param  exec_kind   The target execution kind (cpu or gpu).
 * \param  invocable   The invocable to execute.
 * \param  args        The arguments for the invocable.
 * \tparam Kind        The kind of the exection.
 * \tparam Invocable   The type of the invocable.
 * \tparam Args        The type of the arguments.
 */
template <
  ExecutionKind Kind,
  typename Invocable,
  typename... Args,
  std::enable_if_t<Kind == ExecutionKind::gpu, int> = 0>
auto invoke_generic(
  Execution<Kind> exec_kind, Invocable&& invocable, Args&&... args) noexcept
  -> void {
  kernel::gpu::invoke_generic_impl(
    ripple_forward(invocable), ripple_forward(args)...);
  return;
}

template <
  ExecutionKind Kind,
  typename Invocable,
  typename... Args,
  std::enable_if_t<Kind == ExecutionKind::cpu, int> = 0>
auto invoke_generic(
  Execution<Kind> exec_kind, Invocable&& invocable, Args&&... args) noexcept
  -> void {
  kernel::cpu::invoke_generic_impl(
    ripple_forward(invocable), ripple_forward(args)...);
  return;
}

/**
 * Perfoms an invocation of the invocable, forwarding and possibly manipulating
 * the passed arguments.
 *
 * This will look at the types of the arguments, and for any which are Block
 * types, or BlockEnabled, will pass them as offset iterators to the invocable.
 *
 * If any of the arguments are wrapped with shared wrapper types, they are
 * passed as iterators over shared memory.
 *
 * \note This overload is for specifying the execution kind at runtime.
 *
 * \param  exec_kind   The target execution kind (cpu or gpu).
 * \param  invocable   The invocable to execute.
 * \param  args        The arguments for the invocable.
 * \tparam Invocable   The type of the invocable.
 * \tparam Args        The type of the arguments.
 */
template <typename Invocable, typename... Args>
auto invoke_generic(
  ExecutionKind exec_kind, Invocable&& invocable, Args&&... args) noexcept
  -> void {
  switch (exec_kind) {
    case ExecutionKind::gpu:
      invoke_generic(
        GpuExecutor(), ripple_forward(invocable), ripple_forward(args)...);
      break;
    case ExecutionKind::cpu:
      invoke_generic(
        CpuExecutor(), ripple_forward(invocable), ripple_forward(args)...);
      break;
    default: assert(false && "Invalid execution ");
  }
}

//==--- [simple invoke] ----------------------------------------------------==//

/**
 * This forwards the callable and the block to the cpp implemtation to invoke
 * the callable on each element of the block.
 *
 * This will pass an iterator to the block as the first argument to the
 *  callable, where the iterator has been offset to index i,[j,[k]] in the
 * block.
 *
 * \note This overload is for host blocks, and will run on the CPU.
 *
 * \param  block     The block to invoke the callable on.
 * \param  callabble The callable object.
 * \param  args      Arguments for the callable.
 * \tparam T         The type of the data in the block.
 * \tparam Dims      The number of dimensions in the block.
 * \tparam Callable  The callable object to invoke.
 * \tparam Args      The type of the arguments for the invocation.
 */
template <
  typename T,
  size_t Dims,
  typename Callable,
  typename... Args,
  non_exec_param_enable_t<Callable> = 0>
auto invoke(
  HostBlock<T, Dims>& block, Callable&& callable, Args&&... args) noexcept
  -> void {
  kernel::invoke(block, ripple_forward(callable), ripple_forward(args)...);
}

// clang-format off
/**
 * This forwards the callable and the block to the gpu implemtation to
 * invoke the callable on each element of the block.
 *
 * This will pass an iterator to the block as the first argument to the
 * callable, where the iterator has been offset to index i,[j,[k]] in the
 * block.
 *
 * \note This overload is for device blocks, and will run on the GPU.
 *
 * \param  block     The block to invoke the callable on.
 * \param  callabble The callable object.
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
auto invoke(
  DeviceBlock<T, Dims>& block, Callable&& callable, Args&&... args) noexcept
  -> void {
  // clang-format on
  kernel::gpu::invoke(block, ripple_forward(callable), ripple_forward(args)...);
}

} // namespace ripple

#endif //  RIPPLE_FUNCTIONAL_INVOKE_HPP

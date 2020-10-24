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
#include "kernel/invoke_on_block_.hpp"
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
template <ExecutionKind Kind, typename Invocable, typename... Args>
auto invoke_generic(
  Executor<Kind> exec_kind, Invocable&& invocable, Args&&... args) noexcept
  -> void {
  if constexpr (Kind == ExecutionKind::gpu) {
    kernel::gpu::invoke_generic_impl(
      static_cast<Invocable&&>(invocable), static_cast<Args&&>(args)...);
    return;
  }
  if constexpr (Kind == ExecutionKind::cpu) {
    kernel::cpu::invoke_generic_impl(
      static_cast<Invocable&&>(invocable), static_cast<Args&&>(args)...);
    return;
  }
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
        GpuExecutor(),
        static_cast<Invocable&&>(invocable),
        static_cast<Args&&>(args)...);
      break;
    case ExecutionKind::cpu:
      invoke_generic(
        CpuExecutor(),
        static_cast<Invocable&&>(invocable),
        static_cast<Args&&>(args)...);
      break;
    default: assert(false && "Invalid execution ");
  }
}

/**
 * Invokes the invocable on the block, on the gpu.
 *
 * \todo refactor this into invoke_generic.
 *
 * \param  invocable   The invocable to execute.
 * \param  args        The arguments for the invocable.
 * \tparam Invocable   The type of the invocable.
 * \tparam Args        The type of the arguments
 */
template <typename Invocable, typename... Args>
auto block_invoke(Invocable&& invocable, Args&&... args) noexcept -> void {
  kernel::block_invoke(
    static_cast<Invocable&&>(invocable), static_cast<Args&&>(args)...);
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
  kernel::invoke(
    block, static_cast<Callable&&>(callable), static_cast<Args&&>(args)...);
}

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
  typename T,
  size_t Dims,
  typename Callable,
  typename... Args,
  non_exec_param_enable_t<Callable> = 0>
auto invoke(
  DeviceBlock<T, Dims>& block, Callable&& callable, Args&&... args) noexcept
  -> void {
  kernel::gpu::invoke(
    block, static_cast<Callable&&>(callable), static_cast<Args&&>(args)...);
}

/*==--- [invoke with exec params] ------------------------------------------==*/

/**
 * This forwards the callable and the block to the cpp implemtation to invoke
 * the callable on each element of the block, with the execution parameters
 * specified by the params.
 *
 * \note This overload is for host blocks, and will run on the CPU.
 *
 * \param  block       The block to invoke the callable on.
 * \param  exec_params The parameters which define the execution space.
 * \param  callabble   The callable object.
 * \param  args        Arguments for the callable.
 * \tparam T           The type of the data in the block.
 * \tparam Dims        The number of dimensions in the block.
 * \tparam ExecParams  The type of the execution paramters.
 * \tparam Callable    The callable object to invoke.
 * \tparam Args        The type of the arguments for the invocation.
 */
template <
  typename T,
  size_t Dims,
  typename ExecImpl,
  typename Callable,
  typename... Args,
  exec_param_enable_t<ExecImpl> = 0>
auto invoke(
  HostBlock<T, Dims>& block,
  ExecImpl&&          exec_params,
  Callable&&          callable,
  Args&&... args) noexcept -> void {
  kernel::invoke(
    block,
    exec_params,
    static_cast<Callable&&>(callable),
    static_cast<Args&&>(args)...);
}

/**
 * This forwards the callable and the block to the gpu implemtation to
 * invoke the callable on each element of the block, with the execution
 * parameters specified by the params.
 *
 * This will pass an iterator to the block as the first argument to the
 * callable, where the iterator has been offset to index i,[j,[k]] in the
 * block, and the params as the second argument.
 *
 * \note This overload is for device blocks, and will run on the GPU.
 *
 * \param  block       The block to invoke the callable on.
 * \param  exec_params The parameters which define the execution space.
 * \param  callabble   The callable object.
 * \param  args        Arguments for the callable.
 * \tparam T           The type of the data in the block.
 * \tparam Dims        The number of dimensions in the block.
 * \tparam ExecImpl    The type of the execution implementation.
 * \tparam Callable    The callable object to invoke.
 * \tparam Args        The type of the arguments for the invocation.
 */
template <
  typename T,
  size_t Dims,
  typename ExecImpl,
  typename Callable,
  typename... Args,
  exec_param_enable_t<ExecImpl> = 0>
auto invoke(
  DeviceBlock<T, Dims>& block,
  ExecImpl&&            exec_params,
  Callable&&            callable,
  Args&&... args) noexcept -> void {
  kernel::gpu::invoke(
    block,
    exec_params,
    static_cast<Callable&&>(callable),
    static_cast<Args&&>(args)...);
}

} // namespace ripple

#endif //  RIPPLE_FUNCTIONAL_INVOKE_HPP

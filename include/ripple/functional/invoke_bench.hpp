//==--- ripple/functional/invoke_bench.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_bench.hpp
/// \brief This file implements functionality to invoke a functor and benchmark
///        it.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_INVOKE_BENCH_HPP
#define RIPPLE_FUNCTIONAL_INVOKE_BENCH_HPP

#include "kernel/invoke_cpp_.hpp"
#include "kernel/invoke_cuda_.cuh"

namespace ripple {

//==--- [simple] -----------------------------------------------------------==//

/// This forwards the callable and the block to the cpp implemtation to invoke
/// the \p callable on each element of the block.
///
/// This will pass an iterator to the block as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block.
///
/// It will also profile the kernel, filling the \p event struct with information
/// about the kernel
///
/// This overload is for host blocks, and will run on the CPU.
///
/// \param  block     The block to invoke the callable on.
/// \param  event     The event to fill with profiling information.
/// \param  callabble The callable object.
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
auto invoke_bench(
  HostBlock<T, Dims>& block,
  Event&              event,
  Callable&&          callable,
  Args&&...           args
) -> void {
  kernel::bench::invoke(
    block                           ,
    event                           ,
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
}

/// This forwards the \p callable and the \p block to the cuda implemtation to
/// invoke the \p callable on each element of the block.
///
/// This will pass an iterator to the block as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block.
///
/// It will also profile the kernel, filling the \p event struct with information
/// about the kernel
///
/// This overload is for device blocks, and will run on the GPU.
///
/// \param  block     The block to invoke the callable on.
/// \param  event     The event to fill with profiling information.
/// \param  callabble The callable object.
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
auto invoke_bench(
  DeviceBlock<T, Dims>& block   ,
  Event&                event   ,
  Callable&&            callable,
  Args&&...             args
) -> void {
  kernel::cuda::bench::invoke(
    block, event, std::forward<Callable>(callable), std::forward<Args>(args)...
  );
}

//==--- [with exec params] -------------------------------------------------==//

/// This forwards the \p callable and the \p block to the cpp implemtation to
/// invoke the \p callable on each element of the block, with the execution
/// parameters specified by the \p exec_params.
///
/// This will pass an iterator to the block as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block.
///
/// This overload is for host blocks, and will run on the CPU.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params The parameters which define the execution space.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExecImpl    The type of the execution parameters.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExecImpl,
  typename    Callable,
  typename... Args    ,
  exec_param_enable_t<ExecImpl> = 0
>
auto invoke_bench(
  HostBlock<T, Dims>& block      ,
  ExecImpl&&          exec_params,
  Event&              event      ,
  Callable&&          callable   ,
  Args&&...           args
) -> void {
  kernel::bench::invoke(
    block                           ,
    exec_params                     ,
    event                           , 
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
}

/// This forwards the \p callable and the \p block to the cuda implemtation to
/// invoke the \p callable on each element of the block, with the execution
/// parameters specified by the \p exec_params.
///
/// This will pass an iterator to the block as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block.
///
/// This overload is for device blocks, and will run on the GPU.
///
/// \param  block     The block to invoke the callable on.
/// \param  params    The parameters which define the execution space.
/// \param  callabble The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Params    The type of the execution parameters.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExecImpl,
  typename    Callable,
  typename... Args    ,
  exec_param_enable_t<ExecImpl> = 0
>
auto invoke_bench(
  DeviceBlock<T, Dims>& block      ,
  Event&                event      ,
  ExecImpl&&            exec_params,
  Callable&&            callable   ,
  Args&&...             args
) -> void {
  kernel::cuda::bench::invoke(
    block                              ,
    event                              , 
    std::forward<ExecImpl>(exec_params), 
    std::forward<Callable>(callable)   ,
    std::forward<Args>(args)...
  );
}

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_INVOKE_BNECH_HPP

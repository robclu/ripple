//==--- ripple/core/functional/invoke.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
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
#include "kernel/invoke_cuda_.cuh"
#include "kernel/invoke_pipeline_cuda_.cuh"

namespace ripple {

//==--- [pipeline invoke] --------------------------------------------------==//

/// This invokes the \p pipeline on each element of the \p block. 
///
/// This overload is for device blocks and will run each stage of the pipeline
/// on the GPU.
/// 
/// \param  block    The block to invoke the pipeline on.
/// \param  pipeline The pipeline to invoke on the block.
/// \tparam T        The data type for the block.
/// \tparam Dims     The number of dimensions in the block.
/// \tparam Ops      The type of the pipeline operations.
template <typename T, size_t Dims, typename... Ops>
auto invoke(DeviceBlock<T, Dims>& block, const Pipeline<Ops...>& pipeline)
-> void {
  kernel::cuda::invoke(block, pipeline);
}

//==--- [simple invoke] ----------------------------------------------------==//

/// This forwards the callable and the block to the cpp implemtation to invoke
/// the \p callable on each element of the block.
///
/// This will pass an iterator to the block as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block.
///
/// This overload is for host blocks, and will run on the CPU.
///
/// \param  block     The block to invoke the callable on.
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
auto invoke(HostBlock<T, Dims>& block, Callable&& callable, Args&&... args)
-> void {
  kernel::invoke(
    block, std::forward<Callable>(callable), std::forward<Args>(args)...
  );
}

/// This forwards the \p callable and the \p block to the cuda implemtation to
/// invoke the \p callable on each element of the block.
///
/// This will pass an iterator to the block as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block.
///
/// This overload is for device blocks, and will run on the GPU.
///
/// \param  block     The block to invoke the callable on.
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
auto invoke(DeviceBlock<T, Dims>& block, Callable&& callable, Args&&... args)
-> void {
  kernel::cuda::invoke(
    block, std::forward<Callable>(callable), std::forward<Args>(args)...
  );
}

//==--- [invoke with exec params] ------------------------------------------==//

/// This forwards the callable and the block to the cpp implemtation to invoke
/// the \p callable on each element of the block, with the execution parameters
/// specified by the \p params.
///
/// This overload is for host blocks, and will run on the CPU.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params The parameters which define the execution space.
/// \param  callabble   The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExecParams  The type of the execution paramters.
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
auto invoke(
  HostBlock<T, Dims>& block      ,
  ExecImpl&&          exec_params,
  Callable&&          callable   ,
  Args&&...           args
) -> void {
  kernel::invoke(
    block                           ,
    exec_params                     ,
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
}

/// This forwards the \p callable and the \p block to the cuda implemtation to
/// invoke the \p callable on each element of the block, with the execution
/// parameters specified by the \p params.
///
/// This will pass an iterator to the block as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block, and the \p params as the second argument.
///
/// This overload is for device blocks, and will run on the GPU.
///
/// \param  block       The block to invoke the callable on.
/// \param  exec_params The parameters which define the execution space.
/// \param  callabble   The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExecImpl    The type of the execution implementation.
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
auto invoke(
  DeviceBlock<T, Dims>& block      ,
  ExecImpl&&            exec_params,
  Callable&&            callable   ,
  Args&&...             args
) -> void {
  kernel::cuda::invoke(
    block                           ,
    exec_params                     ,
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
}

//==--- [invoke with multiple blocks] --------------------------------------==//

/// Passes an iterator to the \p block_1 as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block. Additionally, passes an iterator to the \p block_2 as the second
/// argument to the callable, offset to index i, [j, [k]] in the block.
///
/// This overload is for device blocks, and will run on the GPU.
///
/// \param  block_1     The first block to invoke the callable on.
/// \param  block_2     An additional block to pass to the callable.
/// \param  callabble   The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T1          The type of the data in the first block.
/// \tparam Dims1       The number of dimensions in the first block.
/// \tparam T2          The type of the data in the second block.
/// \tparam Dims1       The number of dimensions in the first block.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
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
  kernel::cuda::invoke(
    block_1                         ,
    block_2                         ,
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
}

/// Passes an iterator to the \p block_1 as the first argument to the
/// callable, where the iterator has been offset to index i,[j,[k]] in the
/// block. Additionally, passes an iterator to the \p block_2 as the second
/// argument to the callable, offset to index i, [j, [k]] in the block.
///
/// This overload is for host blocks, and will run on the CPU.
///
/// \param  block_1     The first block to invoke the callable on.
/// \param  block_2     An additional block to pass to the callable.
/// \param  callabble   The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T1          The type of the data in the first block.
/// \tparam Dims1       The number of dimensions in the first block.
/// \tparam T2          The type of the data in the second block.
/// \tparam Dims1       The number of dimensions in the first block.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
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
  HostBlock<T1, Dims1>& block_1 ,
  HostBlock<T2, Dims2>& block_2 ,
  Callable&&            callable,
  Args&&...             args
) -> void {
  kernel::invoke(
    block_1                         ,
    block_2                         ,
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
}

} // namespace ripple

#endif //  RIPPLE_FUNCTIONAL_INVOKE_HPP

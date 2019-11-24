//==--- ripple/functional/invoke.hpp ----------------------- -*- C++ -*- ---==//
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

namespace ripple {

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

} // namespace ripple

#endif //  RIPPLE_FUNCTIONAL_INVOKE_HPP

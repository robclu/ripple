//==--- ../kernel/invoke_generic_impl_.hpp ----------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_generic_impl_.hpp
/// \brief This file provides the implementation of a generic invoke function
///        to invoke a callable type with a variadic number of arguments.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENERIC_IMPL__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENERIC_IMPL__HPP

#include "invoke_utils_.cuh"
#include <ripple/core/execution/dynamic_execution_params.hpp>
#include <ripple/core/execution/execution_params.hpp>
#include <ripple/core/execution/execution_size.hpp>
#include <ripple/core/execution/execution_traits.hpp>
#include <ripple/core/execution/synchronize.hpp>
#include <ripple/core/execution/detail/thread_index_impl_.hpp>
#include <ripple/core/execution/thread_index.hpp>

namespace ripple::kernel::cpu {

/**
 * Implementation of generic invoke for the cpu.
 *
 * This will look at the types of the arguments, and for any which are Block
 * types, or BlockEnabled, will pass them as offset iterators to the invocable.
 *
 * If any of the arguments are wrapped with shared wrapper types, they are
 * passed as iterators over a thread-local tiled memory.
 *
 * \param  invocable The invocable to execute on the gpu.
 * \param  args      The arguments for the invocable.
 * \tparam Invocable The type of the invocable.
 * \tparam Args      The type of the args.
 */
template <typename Invocable, typename... Args>
auto invoke_generic_impl(Invocable&& invocable, Args&&... args) noexcept
  -> void {
  // TODO: Add implementation
}

} // namespace ripple::kernel::cpu

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENERIC_IMPL__HPP

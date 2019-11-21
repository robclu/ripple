//==--- ripple/functional/kernel/invoke_cpp_.hpp ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_cpp_.cuh
/// \brief This file implements functionality to invoke a callable object on the
///        host.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CPP__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CPP__HPP

#include <ripple/container/host_block.hpp>
#include <ripple/perf/perf_traits.hpp>
#include <ripple/utility/dim.hpp>
#include <ripple/utility/number.hpp>
#include <chrono>

namespace ripple::kernel {
namespace detail {

/// Implementation struct to invoke a callable over a number of dimensions.
/// \tparam I The index of the dimension to invoke over.
template <std::size_t I>
struct InvokeImpl {
  /// Invokes the \p callable on the \p it iterator with the \p args.
  /// \param  it        The iterator to pass to the callable.
  /// \param  callable  The callable object.
  /// \param  args      Arguments for the callable.
  /// \tparam Iterator  The type of the iterator.
  /// \tparam Callable  The callable object to invoke.
  /// \tparam Args      The type of the arguments for the invocation.
  template <typename Iterator, typename Callable, typename... Args>
  static auto invoke(Iterator&& it, Callable&& callable, Args&&... args)
  -> void {
    constexpr auto dim = Num<I>();
    for (auto i : range(it.size(dim))) {
      if constexpr (I == 1) {
        ::ripple::detail::thread_idx_.y = i;
      }
      if constexpr (I == 2) {
        ::ripple::detail::thread_idx_.z = i;
      }

      InvokeImpl<I - 1>::invoke(
        it.offset(dim, i)               ,
        std::forward<Callable>(callable),
        std::forward<Args>(args)...
      );
    }
    ::ripple::detail::thread_idx_.x = 0;
    ::ripple::detail::thread_idx_.y = 0;
    ::ripple::detail::thread_idx_.z = 0;
  }

  /// Invokes the \p callable on the \p it iterator with the \p args, and the
  /// execution \p params.
  /// \param  it          The iterator to pass to the callable.
  /// \param  exec_params The execution parameters.
  /// \param  callable    The callable object.
  /// \param  args        Arguments for the callable.
  /// \tparam Iterator    The type of the iterator.
  /// \tparam ExecImpl    The type of the execution params.
  /// \tparam Callable    The callable object to invoke.
  /// \tparam Args        The type of the arguments for the invocation.
  template <
    typename    Iterator,
    typename    ExecImpl,
    typename    Callable,
    typename... Args
  >
  static auto invoke(
    Iterator&& it         ,
    ExecImpl&& exec_params,
    Callable&& callable   ,
    Args&&...  args
  ) -> void {
    constexpr auto dim = Num<I>();
    for (auto i : range(it.size(dim))) {
      if constexpr (I == 1) {
        ::ripple::detail::thread_idx_.y = i;
      }
      if constexpr (I == 2) {
        ::ripple::detail::thread_idx_.z = i;
      }

      InvokeImpl<I - 1>::invoke(
        it.offset(dim, i)                  ,
        std::forward<ExecImpl>(exec_params),
        std::forward<Callable>(callable)   ,
        std::forward<Args>(args)...
      );
    }
    ::ripple::detail::thread_idx_.x = 0;
    ::ripple::detail::thread_idx_.y = 0;
    ::ripple::detail::thread_idx_.z = 0;
  }

};

/// Specialization for the last dimension dimension.
template <>
struct InvokeImpl<0> {
  /// Invokes the \p callable on the \p it iterator with the \p args.
  /// \param  it        The iterator to pass to the callable.
  /// \param  callable  The callable object.
  /// \param  args      Arguments for the callable.
  /// \tparam Iterator  The type of the iterator.
  /// \tparam Callable  The callable object to invoke.
  /// \tparam Args      The type of the arguments for the invocation.
  template <typename Iterator, typename Callable, typename... Args>
  static auto invoke(Iterator&& it, Callable&& callable, Args&&... args)
  -> void {
    for (auto i : range(it.size(dim_x))) {
      ::ripple::detail::thread_idx_.x = i;
      callable(it.offset(dim_x, i), std::forward<Args>(args)...);
    }
  }

  /// Invokes the \p callable on the \p it iterator with the \p args and the \p
  /// execution params.
  /// \param  it          The iterator to pass to the callable.
  /// \param  exec_params The execution params.
  /// \param  callable    The callable object.
  /// \param  args        Arguments for the callable.
  /// \tparam Iterator    The type of the iterator.
  /// \tparam Params      The type of the execution parameters.
  /// \tparam Callable    The callable object to invoke.
  /// \tparam Args        The type of the arguments for the invocation.
  template <
    typename    Iterator,
    typename    ExecImpl,
    typename    Callable,
    typename... Args
  >
  static auto invoke(
    Iterator&& it     ,
    ExecImpl&& params,
    Callable&& callable,
    Args&&...  args
  ) -> void {
    for (auto i : range(it.size(dim_x))) {
      ::ripple::detail::thread_idx_.x = i;
      if constexpr (ExecTraits<ExecImpl>::uses_shared) {
        auto iter = it.offset(dim_x, i);
        callable(iter, iter, std::forward<Args>(args)...);
      } else {
        callable(it.offset(dim_x, i), params, std::forward<Args>(args)...);
      }
    }
  }
};

} // namespace detail

//==--- [invoke simple] ----------------------------------------------------==//

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
auto invoke(HostBlock<T, Dims>& block, Callable&& callable, Args&&... args)
-> void {
  detail::InvokeImpl<Dims - 1>::invoke(
    block.begin(),
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );   
}

//==--- [execution params invoke] ------------------------------------------==//

/// Invokes the \p callale on each element in the \p block, using the \p params
/// execution params to order the invocation.
///
/// \param  block     The block to invoke the callable on.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExecImpl,
  typename    Callable,
  typename... Args    ,
  non_event_enable_t<Callable> = 0
>
auto invoke(
  HostBlock<T, Dims>& block      ,
  ExecImpl&&          exec_params,
  Callable&&          callable   ,
  Args&&...           args
) -> void {
  detail::InvokeImpl<Dims - 1>::invoke(
    block.begin()                      ,
    std::forward<ExecImpl>(exec_params),
    std::forward<Callable>(callable)   ,
    std::forward<Args>(args)...
  );   
}

namespace bench {

/// Invokes the \p callale on each element in the \p block, profiling the kernel
/// and filling the \p event with profiling information.
///
/// \param  block     The block to invoke the callable on.
/// \param  event     The event for profiling information.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T         The type of the data in the block.
/// \tparam Dims      The number of dimensions in the block.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <typename T, std::size_t Dims, typename Callable, typename... Args>
auto invoke(
  HostBlock<T, Dims>& block,
  Event&              event,
  Callable&&          callable,
  Args&&...           args
) -> void {
  using clock_t = std::chrono::high_resolution_clock;

  auto start = clock_t::now();
  detail::InvokeImpl<Dims - 1>::invoke(
    block.begin(),
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );
  auto end = clock_t::now();
  event.elapsed_time_ms = 
    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
    / 1000000.0f;
}

/// Invokes the \p callale on each element in the \p block, profiling the kernel
/// and filling the \p event with profiling information.
///
/// \param  block       The block to invoke the callable on.
/// \param  event       The event for profiling information.
/// \param  exec_params The execution paramters for the invoke.
/// \param  callable    The callable object.
/// \param  args        Arguments for the callable.
/// \tparam T           The type of the data in the block.
/// \tparam Dims        The number of dimensions in the block.
/// \tparam ExecImpl    The implementation type of the execution paramters.
/// \tparam Callable    The callable object to invoke.
/// \tparam Args        The type of the arguments for the invocation.
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    ExecImpl,
  typename    Callable,
  typename... Args>
auto invoke(
  HostBlock<T, Dims>& block      ,
  ExecImpl&&          exec_params,
  Event&              event      ,
  Callable&&          callable   ,
  Args&&...           args
) -> void {
  using clock_t = std::chrono::high_resolution_clock;

  auto start = clock_t::now();
  detail::InvokeImpl<Dims - 1>::invoke(
    block.begin()                      ,
    std::forward<ExecImpl>(exec_params),
    std::forward<Callable>(callable)   ,
    std::forward<Args>(args)...
  );
  auto end = clock_t::now();
  event.elapsed_time_ms = 
    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
    / 1000000.0f;
}

} // namespace bench

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CPP__HPP

//==--- ripple/core/functional/kernel/invoke_cpp_.hpp ----------- -*- C++ -*- ---==//
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

#include <ripple/core/container/host_block.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>
#include <ripple/core/perf/perf_traits.hpp>
#include <ripple/core/utility/dim.hpp>
#include <ripple/core/utility/number.hpp>
#include <chrono>

namespace ripple::kernel {
namespace detail {

//==--- [invoke blocked implementation] ------------------------------------==//

/// Implementation struct to invoke a callable on a block.
/// \tparam I The index of the dimension to invoke for.
template <std::size_t I>
struct InvokeOnBlock {
  /// Invokes the \p callable on the \p it iterator with the \p args, and the
  /// execution \p exec_params, in a blocked manner.
  /// \param  it          The iterator to pass to the callable.
  /// \param  exec_params The execution parameters.
  /// \param  threads     The number fo threads in each block.
  /// \param  blocks      The number of blocks.
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
    typename... Args    ,
    exec_param_enable_t<ExecImpl> = 0
  >
  static auto invoke(
    Iterator&& it         ,
    ExecImpl&& exec_params,
    dim3&      threads    ,
    Callable&& callable   ,
    Args&&...  args
  ) -> void {
    using namespace ::ripple::detail;
    constexpr auto dim        = Num<I>();
    const auto block_threads  = I == 1 ? threads.y  : threads.z;
    const auto initial_offset = 
      block_threads * (I == 1 ? block_idx_.y : block_idx_.z);

    for (auto i : range(block_threads)) {
      if constexpr (I == 1) {
        thread_idx_.y = i;
        if (thread_idx_.y + initial_offset > it.size(dim) - 1) {
          return;
        }
      }
      if constexpr (I == 2) {
        thread_idx_.z = i;
        if (thread_idx_.z + initial_offset > it.size(dim) - 1) {
          return;
        }
      }

      InvokeOnBlock<I - 1>::invoke(
        it.offset(dim, i)                  ,
        std::forward<ExecImpl>(exec_params),
        threads                            ,
        std::forward<Callable>(callable)   ,
        std::forward<Args>(args)...
      );
    }
  }
};

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <>
struct InvokeOnBlock<0> {
  /// Invokes the \p callable on the \p it iterator with the \p args, and the
  /// execution \p exec_params, in a blocked manner.
  /// \param  it          The iterator to pass to the callable.
  /// \param  exec_params The execution parameters.
  /// \param  threads     The number fo threads in each block.
  /// \param  blocks      The number of blocks.
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
    typename... Args    ,
    exec_param_enable_t<ExecImpl> = 0
  >
  static auto invoke(
    Iterator&& it         ,
    ExecImpl&& exec_params,
    dim3&      threads    ,
    Callable&& callable   ,
    Args&&...  args
  ) -> void {
    using namespace ::ripple::detail;
    const auto block_threads  = threads.x;
    const auto initial_offset = threads.x * block_idx_.x;
    
    for (auto i : range(block_threads)) {
      thread_idx_.x = i;
      if (thread_idx_.x + initial_offset > it.size(dim_x) - 1) {
        return;
      }

      callable(it.offset(dim_x, i), std::forward<Args>(args)...);
    }
  }
};


//==--- [invoke blocked implementation] ------------------------------------==//

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <std::size_t I>
struct InvokeBlockedImpl {
  /// Invokes the \p callable on the \p it iterator with the \p args, and the
  /// execution \p exec_params, in a blocked manner.
  /// \param  it          The iterator to pass to the callable.
  /// \param  exec_params The execution parameters.
  /// \param  threads     The number fo threads in each block.
  /// \param  blocks      The number of blocks.
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
    typename... Args    ,
    exec_param_enable_t<ExecImpl> = 0
  >
  static auto invoke(
    Iterator&& it         ,
    ExecImpl&& exec_params,
    dim3&      threads    ,
    dim3&      blocks     ,
    Callable&& callable   ,
    Args&&...  args
  ) -> void {
    using namespace ::ripple::detail;
    constexpr auto dim    = Num<I>();
    const auto dim_blocks = I == 1 ? blocks.y  : blocks.z;
    const auto dim_offset = I == 1 ? threads.y : threads.z;

    for (auto i : range(dim_blocks)) {
      if constexpr (I == 1) {
        block_idx_.y = i;
      }
      if constexpr (I == 2) {
        block_idx_.z = i;
      }

      InvokeBlockedImpl<I - 1>::invoke(
        it.offset(dim, i * dim_offset)     ,
        std::forward<ExecImpl>(exec_params),
        threads                            ,
        blocks                             ,
        std::forward<Callable>(callable)   ,
        std::forward<Args>(args)...
      );
    }
  }
};

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <>
struct InvokeBlockedImpl<0> {
  /// Invokes the \p callable on the \p it iterator with the \p args, and the
  /// execution \p exec_params, in a blocked manner.
  /// \param  it          The iterator to pass to the callable.
  /// \param  exec_params The execution parameters.
  /// \param  threads     The number fo threads in each block.
  /// \param  blocks      The number of blocks.
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
    typename... Args    ,
    exec_param_enable_t<ExecImpl> = 0
  >
  static auto invoke(
    Iterator&& it         ,
    ExecImpl&& exec_params,
    dim3&      threads    ,
    dim3&      blocks     ,
    Callable&& callable   ,
    Args&&...  args
  ) -> void {
    using namespace ::ripple::detail;
    const auto dim_blocks = blocks.x;
    const auto dim_offset = threads.x;
    
    constexpr auto dims = iterator_traits_t<Iterator>::dimensions;

    for (auto i : range(dim_blocks)) {
      block_idx_.x = i;

      InvokeOnBlock<dims - 1>::invoke(
        it.offset(dim_x, i * dim_offset)   ,
        std::forward<ExecImpl>(exec_params),
        threads                            ,
        std::forward<Callable>(callable)   ,
        std::forward<Args>(args)...
      );
    }
  }
};

//==--- [invoke implementation] --------------------------------------------==//

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
    typename... Args    ,
    exec_param_enable_t<ExecImpl> = 0
  >
  static auto invoke_exec_params(
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

      InvokeImpl<I - 1>::invoke_exec_params(
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

  //==--- [multiple blocks] ------------------------------------------------==//
  
  /// Invokes the \p callable passing the \p it_1 iterator as the first
  /// argument, and the \p it_2 iterator as the second argument, forwarding
  /// the \p args.
  ///
  /// \param  it_1       The first iterator to pass to the callable.
  /// \param  it_2       The second iterator to pass to the callable.
  /// \param  callable   The callable object.
  /// \param  args       Arguments for the callable.
  /// \tparam Iterator1  The type of the first iterator.
  /// \tparam Iterator2  The type of the second iterator.
  /// \tparam Callable   The callable object to invoke.
  /// \tparam Args       The type of the arguments for the invocation.
  template <
    typename    Iterator1,
    typename    Iterator2,
    typename    Callable ,
    typename... Args     ,
    non_exec_param_enable_t<Iterator2> = 0
  >
  static auto invoke_multi(
    Iterator1&& it_1    , 
    Iterator2&& it_2    , 
    Callable&&  callable,
    Args&&...   args
  ) -> void {
    constexpr auto dim = Num<I>();
    for (auto i : range(it_1.size(dim))) {
      if constexpr (I == 1) {
        ::ripple::detail::thread_idx_.y = i;
      }
      if constexpr (I == 2) {
        ::ripple::detail::thread_idx_.z = i;
      }

      // Need to make sure that if the second iterator is of a lower dimension,
      // that we dont try to offset into an invalid dimension.
      InvokeImpl<I - 1>::invoke_multi(
        it_1.offset(dim, i)                                 ,
        dim < it_2.dimensions() ? it_2.offset(dim, i) : it_2,
        std::forward<Callable>(callable)                    ,
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
    typename... Args    ,
    exec_param_enable_t<ExecImpl> = 0
  >
  static auto invoke_exec_params(
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

  //==--- [multiple blocks] ------------------------------------------------==//
  
  /// Invokes the \p callable passing the \p it_1 iterator amd the \p it_2
  /// iterator as the first arguments, forwarding the \p args.
  /// \param  it_1       The first iterator to pass to the callable.
  /// \param  it_2       The second iterator to pass to the callable.
  /// \param  callable   The callable object.
  /// \param  args       Arguments for the callable.
  /// \tparam Iterator1  The type of the first iterator.
  /// \tparam Iterator2  The type of the second iterator.
  /// \tparam Callable   The callable object to invoke.
  /// \tparam Args       The type of the arguments for the invocation.
  template <
    typename    Iterator1,
    typename    Iterator2,
    typename    Callable ,
    typename... Args     ,
    non_exec_param_enable_t<Iterator2> = 0
  >
  static auto invoke_multi(
    Iterator1&& it_1    ,
    Iterator2&& it_2    ,
    Callable&&  callable,
    Args&&...   args
  ) -> void {
    for (auto i : range(it_1.size(dim_x))) {
      ::ripple::detail::thread_idx_.x = i;
      callable(
        it_1.offset(dim_x, i),
        it_2.offset(dim_x, i),
        std::forward<Args>(args)...
      );
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
template <
  typename    T       ,
  std::size_t Dims    ,
  typename    Callable, 
  typename... Args    ,
  non_exec_param_enable_t<Callable> = 0
>
auto invoke(HostBlock<T, Dims>& block, Callable&& callable, Args&&... args)
-> void {
  using namespace ::ripple::detail;
  block_idx_.x  = 0;
  block_idx_.y  = 0;
  block_idx_.z  = 0;
  thread_idx_.x = 0;
  thread_idx_.y = 0;
  thread_idx_.z = 0;

  block_dim_.x = block.size(dim_x);
  block_dim_.y = Dims >= 2 ? block.size(dim_y) : 1;
  block_dim_.z = Dims >= 3 ? block.size(dim_z) : 1;

  detail::InvokeImpl<Dims - 1>::invoke(
    block.begin(),
    std::forward<Callable>(callable),
    std::forward<Args>(args)...
  );   

  block_idx_.x  = 0;
  block_idx_.y  = 0;
  block_idx_.z  = 0;
  thread_idx_.x = 0;
  thread_idx_.y = 0;
  thread_idx_.z = 0;

  block_dim_.x = 1;
  block_dim_.y = 1;
  block_dim_.z = 1;
}

/// Invokes the \p callale on each element in \p block_1, additionally passing
/// an iterator to each element in \p block_2.
///
/// \param  block_1   The first block to pass to the callable.
/// \param  block_2   The second block to pass to the callable.
/// \param  callable  The callable object.
/// \param  args      Arguments for the callable.
/// \tparam T1        The type of the data in the first block.
/// \tparam Dims1     The number of dimensions in the first block.
/// \tparam T2        The type of the data in the second block.
/// \tparam Dims2     The number of dimensions in the second block.
/// \tparam Callable  The callable object to invoke.
/// \tparam Args      The type of the arguments for the invocation.
template <
  typename    T1      ,
  std::size_t Dims1   ,
  typename    T2      ,
  std::size_t Dims2   , 
  typename    Callable, 
  typename... Args    ,
  non_exec_param_enable_t<T1> = 0
>
auto invoke(
  HostBlock<T1, Dims1>& block_1 , 
  HostBlock<T2, Dims2>& block_2 ,
  Callable&&            callable,
  Args&&...             args
) -> void {
  detail::InvokeImpl<Dims1 - 1>::invoke_multi(
    block_1.begin()                 ,
    block_2.begin()                 ,
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
  exec_param_enable_t<ExecImpl> = 0
>
auto invoke(
  HostBlock<T, Dims>& block      ,
  ExecImpl&&          exec_params,
  Callable&&          callable   ,
  Args&&...           args
) -> void {
  auto [threads, blocks] = get_exec_size(block, exec_params);

  using namespace ::ripple::detail;

  block_dim_.x  = threads.x;
  block_dim_.y  = threads.y;
  block_dim_.z  = threads.z;
  grid_dim_.x   = blocks.x;
  grid_dim_.y   = blocks.y;
  grid_dim_.z   = blocks.z;
  thread_idx_.x = 0;
  thread_idx_.y = 0;
  thread_idx_.z = 0;

  detail::InvokeBlockedImpl<Dims - 1>::invoke(
    block.begin()                      ,
    std::forward<ExecImpl>(exec_params),
    threads                            ,
    blocks                             ,
    std::forward<Callable>(callable)   ,
    std::forward<Args>(args)...
  );

  block_dim_.x = 1;
  block_dim_.y = 1;
  block_dim_.z = 1;
  grid_dim_.x  = 1;
  grid_dim_.y  = 1;
  grid_dim_.z  = 1;
}

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_CPP__HPP

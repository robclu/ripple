/**=--- ripple/kernel/detail/invoke_generic_impl_.hpp ----- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  invoke_generic_impl_.hpp
 * \brief This file implements functionality to invoke a callable on variadic
 *        arguments on the cpu.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENRIC__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENRIC__HPP

#include "invoke_utils_.hpp"
#include <ripple/algorithm/max_element.hpp>

namespace ripple::kernel::cpu {

/**
 * Gets an iterator from the host block.
 *
 * \note This overload is for host blocks.
 *
 * \param  block The block to get the iterator from.
 * \tparam T     The type of the block.
 */
template <typename T, block_enable_t<T> = 0>
auto get_iterator(T&& host_block) noexcept {
  return host_block.begin();
}

/**
 * Gets an iterator from the block.
 *
 * \note This overload is for block types.
 *
 * \param block The blocks to get the iterator from.
 */
template <typename T, multiblock_enable_t<T> = 0>
auto get_iterator(T&& block) noexcept {
  return block.host_iterator();
}

/*
template <typename T>
using block_shared_enable_t =
  std::enable_if_t<is_shared_wrapper_v<T> && is_any_block_v<T>, int>;

template <typename T>
using block_non_shared_enable_t =
  std::enable_if_t<!is_shared_wrapper_v<T> && is_any_block_v<T>, int>;

template <typename T>
using non_block_shared_enable_t =
  std::enable_if_t<is_shared_wrapper_v<T> && !is_any_block_v<T>, int>;
*/

template <typename T>
using non_any_block_or_shared_or_expansion_enable_t = std::enable_if_t<
  !is_any_block_v<T> && !is_shared_wrapper_v<T> && !is_expansion_wrapper_v<T>,
  int>;

/**
 * Overload of iterator offsetting for the case that the input type is a block.
 * This gets an iterator from the block and offsets it using the offsets.
 *
 * \note This overload is for block enabled types.
 *
 * \param  block     The block type to get the iterator from.
 * \param  offsets   The offsets for the iterator.
 * \tparam T         The type of the non-block.
 * \tparam Offsets   The type of the offsets.
 */
template <typename T, typename... Offsets, any_block_enable_t<T> = 0>
decltype(auto) shift_if_iterator(T&& block, Offsets&&... offsets) noexcept {
  auto iter = get_iterator(block);
  auto offs = make_tuple(ripple_forward(offsets)...);
  unrolled_for<sizeof...(Offsets)>(
    [&](auto i) { iter.shift(Dimension<i>(), get<i>(offs)); });
  return iter;
}

/**
 * Overload of iterator offsetting for the case that the input type is a block
 * and it is wrapped for shared memory.
 *
 * \note This overload is for block enabled types.
 *
 * \param  wrapper   The wrapper of the block data.
 * \param  offsets   The offsets for the iterator.
 * \tparam T         The type of the non-block.
 * \tparam Offsets   The type of the offsets.
 */
template <typename T, typename... Offsets, any_block_enable_t<T> = 0>
decltype(auto)
shift_if_iterator(SharedWrapper<T>& wrapper, Offsets&&... offsets) noexcept {
  auto iter = get_iterator(wrapper.wrapped);
  auto offs = make_tuple(ripple_forward(offsets)...);
  unrolled_for<sizeof...(Offsets)>(
    [&](auto i) { iter.shift(Dimension<i>(), get<i>(offs)); });
  return iter;
}

template <typename T, typename... Offsets, any_block_enable_t<T> = 0>
decltype(auto)
shift_if_iterator(ExpansionWrapper<T>& wrapper, Offsets&&... offsets) noexcept {
  // TODO: Add expansion factor ...
  auto iter = get_iterator(wrapper.wrapped);
  auto offs = make_tuple(ripple_forward(offsets)...);
  unrolled_for<sizeof...(Offsets)>(
    [&](auto i) { iter.shift(Dimension<i>(), get<i>(offs)); });
  return iter;
}

/**
 * Overload of iterator offsetting for the case that the input type is not
 * a block. This just forwards the input type back.
 *
 * \note This overload is only for non-block enabled types.
 *
 * \param  non_block The non block type to return.
 * \param  offsets   The offsets for the iterator.
 * \tparam T         The type of the non-block.
 * \tparam Offsets   The type of the offsets.
 */
template <
  typename T,
  typename... Offsets,
  non_any_block_or_shared_or_expansion_enable_t<T> = 0>
decltype(auto) shift_if_iterator(T&& non_block, Offsets&&... offsets) noexcept {
  return ripple_forward(non_block);
}

/**
 * Overload of iterator shifting for a shared wrapper which does not hold a
 * block type.
 *
 * \note This overload is for non-block-enabled types.
 *
 * \param  wrapper The wrapper of the data.
 * \param  offets  The offsets for the iterator.
 * \tparam T       The type being wrapped.
 * \tparam Offsets The type of the offsets.
 */
template <typename T, typename... Offsets, non_any_block_enable_t<T> = 0>
decltype(auto)
shift_if_iterator(SharedWrapper<T>& wrapper, Offsets&&... offsets) noexcept {
  return wrapper.wrapped;
}

template <typename T, typename... Offsets, non_any_block_enable_t<T> = 0>
decltype(auto)
shift_if_iterator(ExpansionWrapper<T>& wrapper, Offsets&&... offsets) noexcept {
  return wrapper.wrapped;
}

/**
 * Generic invoke implementation struct.
 */
template <size_t>
struct InvokeGenericImpl {};

/**
 * Implementation struct to invoke a callable in a blocked manner,
 * \tparam I The index of the dimension to invoke for.
 */
template <>
struct InvokeGenericImpl<2> {
  /**
   * Invokes the callable on the iterator with the given args and execution
   *  params
   *  \param  it          The iterator to pass to the callable.
   *  \param  exec_params The execution parameters.
   *  \param  threads     The number fo threads in each block.
   *  \param  blocks      The number of blocks.
   *  \param  callable    The callable object.
   *  \param  args        Arguments for the callable.
   *  \tparam Iterator    The type of the iterator.
   *  \tparam ExecImpl    The type of the execution params.
   *  \tparam Callable    The callable object to invoke.
   *  \tparam Args        The type of the arguments for the invocation.
   */
  template <typename Callable, typename... Args>
  static auto
  invoke(Callable&& callable, const DimSizes& sizes, Args&&... args) -> void {
    for (size_t k = 0; k < sizes[dimz()]; ++k) {
      for (size_t j = 0; j < sizes[dimy()]; ++j) {
        for (size_t i = 0; i < sizes[dimx()]; ++i) {
          callable(shift_if_iterator(ripple_forward(args), i, j, k)...);
        }
      }
    }
  }
};

/**
 * Implementation struct to invoke a callable in a blocked manner,
 * \tparam I The index of the dimension: to invoke for.
 */
template <>
struct InvokeGenericImpl<1> {
  /**
   * Invokes the callable on the iterator with the given args and execution
   * params.
   *  \param  it          The iterator to pass to the callable.
   *  \param  exec_params The execution parameters.
   *  \param  threads     The number fo threads in each block.
   *  \param  blocks      The number of blocks.
   *  \param  callable    The callable object.
   *  \param  args        Arguments for the callable.
   *  \tparam Iterator    The type of the iterator.
   *  \tparam ExecImpl    The type of the execution params.
   *  \tparam Callable    The callable object to invoke.
   *  \tparam Args        The type of the arguments for the invocation.
   */
  template <typename Callable, typename... Args>
  static auto
  invoke(Callable&& callable, const DimSizes& sizes, Args&&... args) -> void {
    for (size_t j = 0; j < sizes[dimy()]; ++j) {
      for (size_t i = 0; i < sizes[dimx()]; ++i) {
        callable(shift_if_iterator(ripple_forward(args), i, j)...);
      }
    }
  }
};

/**
 * Implementation struct to invoke a callable in a blocked manner,
 * \tparam I The index of the dimension to invoke for.
 */
template <>
struct InvokeGenericImpl<0> {
  /**
   * Invokes the callable on the  iterator with the given args and execution
   * params.
   *  \param  it          The iterator to pass to the callable.
   *  \param  exec_params The execution parameters.
   *  \param  threads     The number fo threads in each block.
   *  \param  blocks      The number of blocks.
   *  \param  callable    The callable object.
   *  \param  args        Arguments for the callable.
   *  \tparam Iterator    The type of the iterator.
   *  \tparam ExecImpl    The type of the execution params.
   *  \tparam Callable    The callable object to invoke.
   *  \tparam Args        The type of the arguments for the invocation.
   */
  template <typename Callable, typename... Args>
  static auto
  invoke(Callable&& callable, const DimSizes& sizes, Args&&... args) -> void {
    for (size_t i = 0; i < sizes[dimx()]; ++i) {
      callable(shift_if_iterator(ripple_forward(args), i)...);
    }
  }
};

/**
 * Implementation of generic invoke for the cpu.
 *
 * This will look at the types of the arguments, and for any which are Block
 * types, or BlockEnabled, will pass them as offset iterators to the
 * invocable.
 *
 * If any of the arguments are wrapped with shared wrapper types, they are
 * passed as iterators over *tiled* memory, which is thread-local storage.
 *
 * \param  invocable The invocable to execute on the gpu.
 * \param  args      The arguments for the invocable.
 * \tparam Invocable The type of the invocable.
 * \tparam Args      The type of the args.
 */
template <typename Invocable, typename... Args>
auto invoke_generic_impl(Invocable&& invocable, Args&&... args) noexcept
  -> void {
  constexpr size_t dims = max_element(any_block_traits_t<Args>::dimensions...);

  // Find the grid sizes:
  const auto sizes = DimSizes{
    max_element(size_t{1}, get_size_if_block(args, dimx())...),
    max_element(size_t{1}, get_size_if_block(args, dimy())...),
    max_element(size_t{1}, get_size_if_block(args, dimz())...)};

  InvokeGenericImpl<dims - 1>::invoke(
    ripple_forward(invocable), sizes, ripple_forward(args)...);
}

} // namespace ripple::kernel::cpu

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENERIC_IMPL__HPP
//==--- ../kernel/invoke_generic_impl_.hpp ----------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_generic_impl_.hpp
/// \brief This file implements functionality to invoke a callable on variadic
///        arguments on the cpu.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENRIC__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENRIC__HPP

#include "invoke_utils.hpp"
#include <ripple/core/algorithm/max_element.hpp>

namespace ripple::kernel::cpu {

/*==--- [iterator offsetting] ----------------------------------------------==*/

/**
 * Overload of iterator offsetting for the case that the input type is a block.
 * This gets an iterator from the block and offsets it using the offsets.
 *
 * \param  block     The block type to get the iterator from.
 * \param  offsets   The offsets for the iterator.
 * \tparam T         The type of the non-block.
 * \tparam Offsets   The type of the offsets.
 */

// template <typename T, typename... Offsets, block_enabled_t<T> = 0>
// auto shift_if_iterator(T& block, Offsets&&... offsets) noexcept
//  -> decltype(block.begin()) {
//  auto iter = block.begin();
//  auto offs = make_tuple(static_cast<Offsets&&>(offsets)...);
//  unrolled_for<sizeof...(Offsets)>(
//    [&](auto i) { iter.shift(Dimension<i>(), get<i>(offs)); });
//  return iter;
//}

/**
 * Overload of iterator offsetting for the case that the input type is not
 * a block. This just forwards the input type back.
 *
 * \param  non_block The non block type to return.
 * \param  offsets   The offsets for the iterator.
 * \tparam T         The type of the non-block.
 * \tparam Offsets   The type of the offsets.
 */
// template <typename T, typename... Offsets, non_block_enabled_t<T> = 0>
// decltype(auto) shift_if_iterator(T&& non_block, Offsets&&... offsets)
// noexcept {
//  return static_cast<T&&>(non_block);
//}
//

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
template <typename T, non_block_enable_t<T> = 0>
auto get_iterator(T&& block) noexcept {
  return block.host_iterator();
}

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
template <typename T, typename... Offsets, block_enabled_t<T> = 0>
auto shift_if_iterator(T&& block, Offsets&&... offsets) noexcept {
  auto iter = get_iterator(block);
  auto offs = make_tuple(static_cast<Offsets&&>(offsets)...);
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
template <typename T, typename... Offsets, block_enabled_t<T> = 0>
auto shift_if_iterator(
  SharedWrapper<T>& wrapper, Offsets&&... offsets) noexcept {
  auto iter = get_iterator(wrapper.wrapped);
  auto offs = make_tuple(static_cast<Offsets&&>(offsets)...);
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
template <typename T, typename... Offsets, non_block_enabled_t<T> = 0>
decltype(auto) shift_if_iterator(T&& non_block, Offsets&&... offsets) noexcept {
  return static_cast<T&&>(non_block);
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
template <typename T, typename... Offsets, non_block_enabled_t<T> = 0>
decltype(auto)
shift_if_iterator(SharedWrapper<T>&& wrapper, Offsets&&... offsets) noexcept {
  return wrapper.wrapped;
}

/**
 * Generic invoke implementation struct.
 */
template <size_t>
struct InvokeGenericImpl {};

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <>
struct InvokeGenericImpl<2> {
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
  template <typename Callable, typename... Args>
  static auto
  invoke(Callable&& callable, const DimSizes& sizes, Args&&... args) -> void {
    for (size_t k = 0; k < sizes[dim_z]; ++k) {
      for (size_t j = 0; j < sizes[dim_y]; ++j) {
        for (size_t i = 0; i < sizes[dim_x]; ++i) {
          callable(shift_if_iterator(static_cast<Args&&>(args), i, j, k)...);
        }
      }
    }
  }
};

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <>
struct InvokeGenericImpl<1> {
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
  template <typename Callable, typename... Args>
  static auto
  invoke(Callable&& callable, const DimSizes& sizes, Args&&... args) -> void {
    //    Tuple<Args...> t(static_cast<Args&&>(args)...);
    for (size_t j = 0; j < sizes[dim_y]; ++j) {
      for (size_t i = 0; i < sizes[dim_x]; ++i) {
        callable(shift_if_iterator(static_cast<Args&&>(args), i, j)...);
      }
    }
  }
};

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <>
struct InvokeGenericImpl<0> {
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
  template <typename Callable, typename... Args>
  static auto
  invoke(Callable&& callable, const DimSizes& sizes, Args&&... args) -> void {
    for (size_t i = 0; i < sizes[dim_x]; ++i) {
      callable(shift_if_iterator(static_cast<Args&&>(args), i)...);
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
  constexpr size_t dims =
    max_element(block_enabled_traits_t<Args>::dimensions...);

  // Find the grid sizes:
  const auto sizes =
    DimSizes{max_element(size_t{1}, get_size_if_block(args, dim_x)...),
             max_element(size_t{1}, get_size_if_block(args, dim_y)...),
             max_element(size_t{1}, get_size_if_block(args, dim_z)...)};

  InvokeGenericImpl<dims - 1>::invoke(
    static_cast<Invocable&&>(invocable), sizes, static_cast<Args>(args)...);
}

} // namespace ripple::kernel::cpu

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENERIC_IMPL__HPP
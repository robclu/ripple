//==--- ../invoke_on_block_.hpp ---------------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_on_block_.hpp
/// \brief This file implements functionality to invoke a callable on variadic
///        arguments with support for host blocks.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_ON_BLOCK__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_ON_BLOCK__HPP

#include "invoke_utils.hpp"
#include <ripple/core/algorithm/max_element.hpp>

namespace ripple::kernel {

namespace detail {

template <typename T, typename Dim, block_enabled_t<T> = 0>
auto get_block_index(T&& block, Dim&& dim) -> size_t {
  return block.index(std::forward<Dim>(dim));
}

template <typename T, typename Dim, non_block_enabled_t<T> = 0>
auto get_block_index(T&& non_block, Dim&& dim) -> size_t {
  return 0;
}

template <typename T, typename Dim, block_enable_t<T> = 0>
auto offset_iters(T& block, Dim&& dim, int offset) {
  return block.begin().offset(std::forward<Dim>(dim), offset);
}

template <typename T, typename Dim, non_block_enable_t<T> = 0>
auto offset_iters(T&& non_block, Dim&& dim, int offset) noexcept
  -> decltype(std::forward<T>(non_block)) {
  return std::forward<T>(non_block);
}

template <typename T, typename DimA, typename DimB, block_enable_t<T> = 0>
auto offset_iters(
  T& block, DimA&& dim_a, size_t offset_a, DimB&& dim_b, size_t offset_b) {
  return block.begin()
    .offset(std::forward<DimA>(dim_a), offset_a)
    .offset(std::forward<DimB>(dim_b), offset_b);
}

template <typename T, typename DimA, typename DimB, non_block_enable_t<T> = 0>
auto offset_iters(
  T&&    non_block,
  DimA&& dim_a,
  size_t offset_a,
  DimB&& dim_b,
  size_t offset_b) noexcept -> decltype(std::forward<T>(non_block)) {
  return std::forward<T>(non_block);
}

template <size_t>
struct BlockInvoke {};

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <>
struct BlockInvoke<1> {
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
  static auto invoke(Callable&& callable, Args&&... args) -> void {
    using namespace ::ripple::detail;
    // const auto block_idx = max_element(get_block_index(args, dim_x)...);
    const auto block_size_x =
      max_element(util::get_block_size(std::forward<Args>(args), dim_x)...);
    const auto block_size_y =
      max_element(util::get_block_size(std::forward<Args>(args), dim_y)...);

    for (auto j : range(block_size_y)) {
      for (auto i : range(block_size_x)) {
        callable(offset_iters(std::forward<Args>(args), dim_x, i, dim_y, j)...);
      }
    }
  }
};

/// Implementation struct to invoke a callable in a blocked manner,
/// \tparam I The index of the dimension to invoke for.
template <>
struct BlockInvoke<0> {
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
  static auto invoke(Callable&& callable, Args&&... args) -> void {
    using namespace ::ripple::detail;
    // const auto block_idx = max_element(get_block_index(args, dim_x)...);
    const auto block_size =
      max_element(util::get_block_size(std::forward<Args>(args), dim_x)...);

    printf("Block size : %3lu\n", block_size);

    for (auto i : range(block_size)) {
      callable(offset_iters(std::forward<Args>(args), dim_x, i)...);
    }
  }
};

} // namespace detail

template <typename Invocable, typename... Args>
auto block_invoke(Invocable&& invocable, Args&&... args) -> void {
  constexpr size_t dims =
    max_element(block_enabled_traits_t<Args>::dimensions...);

  /*
    // Gets the size of the block:
    const auto exec_params = dynamic_host_params<dims>();

    // Find the grid size:
    const auto sizes = std::array<size_t, 3>{
      max_element(size_t{1}, util::get_block_size(args, dim_x)...),
      max_element(size_t{1}, util::get_block_size(args, dim_y)...),
      max_element(size_t{1}, util::get_block_size(args, dim_z)...)};

    for (auto& s : sizes) {
      printf("Dim size : %3lu\n", s);
    }

    auto [threads, blocks] = util::get_execution_sizes(exec_params, sizes);
  */

  detail::BlockInvoke<dims - 1>::invoke(
    std::forward<Invocable>(invocable), std::forward<Args>(args)...);
}

} // namespace ripple::kernel

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_ON_BLOCK__HPP
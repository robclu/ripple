//==--- ripple/core/functional/kernel/invoke_utils_.hpp ---- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_utils_.hpp
/// \brief This file implements functionality to invoke a pipeline on various
///        container objects on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP

#include <ripple/core/container/shared_wrapper.hpp>
#include <ripple/core/execution/execution_size.hpp>

namespace ripple::kernel::util {

//==---- [execution sizes] -------------------------------------------------==//

/// Gets the number of threads and thread blocks.
/// \param  params     The paramers which define the execution space.
/// \param  sizes      The sizes of the grid.
/// \tparam ExecParams The type of the execution parameters.
template <typename ExecParams>
auto get_execution_sizes(
  const ExecParams& params, const std::array<size_t, 3>& sizes)
  -> std::tuple<dim3, dim3> {
  auto threads = dim3(1, 1, 1), blocks = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(sizes[dim_x], params.size(dim_x));
  threads.y = get_dim_num_threads(sizes[dim_y], params.size(dim_y));
  threads.z = get_dim_num_threads(sizes[dim_z], params.size(dim_z));
  blocks.x  = get_dim_num_blocks(sizes[dim_x], threads.x);
  blocks.y  = get_dim_num_blocks(sizes[dim_y], threads.y);
  blocks.z  = get_dim_num_blocks(sizes[dim_z], threads.z);

  return std::make_tuple(threads, blocks);
}

//==--- [block extraction] -------------------------------------------------==//

template <typename T>
auto extract_device_block(SharedWrapper<T> wrapper) -> SharedWrapper<T> {
  return SharedWrapper<T>{wrapper.wrapped, wrapper.padding};
}

template <typename T>
decltype(auto) extract_device_block(T&& t) {
  return t;
}

//==--- [block size] -------------------------------------------------------==//

template <typename T, typename Dim, block_enabled_t<T> = 0>
auto get_block_size(T&& block, Dim&& dim) -> size_t {
  return block_enabled_traits_t<T>::dimensions > dim
           ? block.size(std::forward<Dim>(dim))
           : size_t{0};
}

template <typename T, typename Dim, block_enable_t<T> = 0>
auto get_block_size(T&& block, Dim&& dim) -> size_t {
  return block_enabled_traits_t<T>::dimensions > dim
           ? block.size(std::forward<Dim>(dim))
           : size_t{0};
}

template <typename T, typename Dim, non_block_enable_t<T> = 0>
auto get_block_size(T&& t, Dim&& dim) -> size_t {
  return 0;
}

template <typename T, typename Dim>
auto get_block_size(SharedWrapper<T>& t, Dim&& dim) -> size_t {
  return get_block_size(t.wrapped, std::forward<Dim>(dim));
}

//==--- [get iter from block] ----------------------------------------------==//

/// Returns an iterator over the block data if \p t is block enabled.
/// \param  t The block enabled type to get an iterator for.
/// \tparam T The type of the block enabled type.
template <typename T, block_enabled_t<T> = 0>
auto block_iter_or_same(T&& t) -> std::decay_t<decltype(t.host_iterator())> {
  return t.begin_host();
}

/// Returns \p t, without modification.
/// \param  t The block enabled type to get an iterator for.
/// \tparam T The type of the block enabled type.
template <typename T, non_block_enabled_t<T> = 0>
auto block_iter_or_same(T&& t) -> std::decay_t<T> {
  return t;
}

} // namespace ripple::kernel::util

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP
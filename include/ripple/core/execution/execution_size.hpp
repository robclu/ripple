//==--- ripple/core/execution/execution_size.hpp ----------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  execution_size.hpp
/// \brief This file contains functionality for computing the size of an
///        execution space for a kernel.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_EXECUTION_SIZE_HPP
#define RIPPLE_EXECUTION_EXECUTION_SIZE_HPP

#include <ripple/core/container/device_block.hpp>
#include <ripple/core/utility/dim.hpp>
#include <ripple/core/utility/type_traits.hpp>
#include <cmath>
#include <tuple>

namespace ripple {

/**
 * Computes the number of blocks given that there are \p elements in the
 * dimension and \p threads in the dimension for the block.
 *
 * \param  elements    The number of elements in the dimension.
 * \param  max_threads The number of threads in the dimension.
 * \tparam T           The type of the elements.
 * \tparam U           The type of the max threads.
 * \return The number of blocks for the dimension.
 */
template <typename T, typename U>
auto get_dim_num_blocks(T elements, U threads) noexcept -> size_t {
  return std::max(
    static_cast<std::size_t>(
      std::ceil(static_cast<double>(elements) / static_cast<double>(threads))),
    std::size_t{1});
}

/**
 * Computes the number of threads given that there are \p elements in the
 * dimension, and \p max_threads maximum number of threads for the dimension.
 * If elements < max_threads, then this returns \p max_threads, otherwise
 * it returns \p elements.
 *
 * \param  elements    The number of elements in the dimension.
 * \param  max_threads The maximum number of threads for the dimension.
 * \tparam T           The type of the elements.
 * \tparam U           The type of the max threads.
 * \return The number of threads.
 */
template <typename T, typename U>
auto get_dim_num_threads(T elements, U max_threads) noexcept -> size_t {
  return std::min(
    static_cast<std::size_t>(elements), static_cast<std::size_t>(max_threads));
}

/**
 * Computes the number of threads and blocks required to cover the space defined
 * by the \p block, with the execution space defined by the \p exec_params.
 *
 * This overload is only enabled if the \p block is one dimensional.
 *
 * \param  block       The block to generate the execution size for.
 * \param  exec_params The execution parameters.
 * \tparam Block       The type of the block.
 * \tparam ExeImpl     The type of the execution parameter implementation.
 * \return A tuple with { num threads (3d), num blocks (3d) }.
 */
template <
  typename Block,
  typename ExeImpl,
  block_enabled_1d_enable_t<Block> = 0>
auto get_exec_size(
  const Block& block, const ExecParams<ExeImpl>& exec_params) noexcept
  -> std::tuple<dim3, dim3> {
  const auto elems_x = block.size(dim_x);

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, exec_params.size(dim_x));
  blocks.x  = get_dim_num_blocks(elems_x, threads.x);

  return std::make_tuple(threads, blocks);
}

/**
 * Computes the number of threads and blocks required to cover the space defined
 * by the \p block, with the execution space defined by the \p exec_params.
 *
 * This overload is only enabled if the \p block is two dimensional.
 *
 * \param  block       The block to generate the execution size for.
 * \param  exec_params The execution parameters.
 * \tparam Block       The type of the block.
 * \tparam ExeImpl     The type of the execution parameter implementation.
 * \return A tuple with { num threads (3d), num blocks (3d) }.
 */
template <
  typename Block,
  typename ExeImpl,
  block_enabled_2d_enable_t<Block> = 0>
auto get_exec_size(
  const Block& block, const ExecParams<ExeImpl>& exec_params) noexcept
  -> std::tuple<dim3, dim3> {
  const auto elems_x = block.size(dim_x);
  const auto elems_y = block.size(dim_y);

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, exec_params.size(dim_x));
  threads.y = get_dim_num_threads(elems_y, exec_params.size(dim_y));
  blocks.x  = get_dim_num_blocks(elems_x, threads.x);
  blocks.y  = get_dim_num_blocks(elems_y, threads.y);

  return std::make_tuple(threads, blocks);
}

/**
 * Computes the number of threads and blocks required to cover the space defined
 * by the \p block, with the execution space defined by the \p exec_params.
 *
 * This overload is only enabled if the \p block is three dimensional.
 *
 * \param  block       The block to generate the execution size for.
 * \param  exec_params The execution parameters.
 * \tparam Block       The type of the block.
 * \tparam ExeImpl     The type of the execution parameter implementation.
 * \return A tuple with { num threads (3d), num blocks (3d) }.
 */
template <
  typename Block,
  typename ExeImpl,
  block_enabled_3d_enable_t<Block> = 0>
auto get_exec_size(const Block& block, const ExecParams<ExeImpl>& exec_params)
  -> std::tuple<dim3, dim3> {
  const auto elems_x = block.size(dim_x);
  const auto elems_y = block.size(dim_y);
  const auto elems_z = block.size(dim_z);

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, exec_params.size(dim_x));
  threads.y = get_dim_num_threads(elems_y, exec_params.size(dim_y));
  threads.z = get_dim_num_threads(elems_z, exec_params.size(dim_z));
  blocks.x  = get_dim_num_blocks(elems_x, threads.x);
  blocks.y  = get_dim_num_blocks(elems_y, threads.y);
  blocks.z  = get_dim_num_blocks(elems_z, threads.z);

  return std::make_tuple(threads, blocks);
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTION_SIZE_HPP

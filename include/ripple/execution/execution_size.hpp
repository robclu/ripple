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

/**=--- ripple/execution/execution_size.hpp ---------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  execution_size.hpp
 * \brief This file contains functionality for computing the size of an
 *        execution space for a kernel.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_EXECUTION_EXECUTION_SIZE_HPP
#define RIPPLE_EXECUTION_EXECUTION_SIZE_HPP

#include <ripple/container/device_block.hpp>
#include <ripple/utility/dim.hpp>
#include <ripple/utility/type_traits.hpp>
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
    static_cast<size_t>(
      std::ceil(static_cast<float>(elements) / static_cast<float>(threads))),
    size_t{1});
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
    static_cast<size_t>(elements), static_cast<size_t>(max_threads));
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
template <typename Block, typename ExeImpl, any_block_1d_enable_t<Block> = 0>
auto get_exec_size(
  const Block&               block,
  const ExecParams<ExeImpl>& exec_params,
  int                        overlap = 0) noexcept -> std::tuple<dim3, dim3> {
  const auto elems_x = block.size(dimx());

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, exec_params.size(dimx()));
  blocks.x  = get_dim_num_blocks(elems_x, threads.x - overlap);

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
template <typename Block, typename ExeImpl, any_block_2d_enable_t<Block> = 0>
auto get_exec_size(
  const Block&               block,
  const ExecParams<ExeImpl>& exec_params,
  int                        overlap = 0) noexcept -> std::tuple<dim3, dim3> {
  const auto elems_x = block.size(dimx());
  const auto elems_y = block.size(dimy());

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, exec_params.size(dimx()));
  threads.y = get_dim_num_threads(elems_y, exec_params.size(dimy()));
  blocks.x  = get_dim_num_blocks(elems_x + overlap, threads.x);
  blocks.y  = get_dim_num_blocks(elems_y + overlap, threads.y);
  // blocks.x  = get_dim_num_blocks(elems_x, threads.x - overlap);
  // blocks.y  = get_dim_num_blocks(elems_y, threads.y - overlap);

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
template <typename Block, typename ExeImpl, any_block_3d_enable_t<Block> = 0>
auto get_exec_size(
  const Block& block, const ExecParams<ExeImpl>& exec_params, int overlap = 0)
  -> std::tuple<dim3, dim3> {
  const size_t elems_x = block.size(dimx());
  const size_t elems_y = block.size(dimy());
  const size_t elems_z = block.size(dimz());

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, exec_params.size(dimx()));
  threads.y = get_dim_num_threads(elems_y, exec_params.size(dimy()));
  threads.z = get_dim_num_threads(elems_z, exec_params.size(dimz()));
  blocks.x  = get_dim_num_blocks(elems_x, threads.x - overlap);
  blocks.y  = get_dim_num_blocks(elems_y, threads.y - overlap);
  blocks.z  = get_dim_num_blocks(elems_z, threads.z - overlap);

  return std::make_tuple(threads, blocks);
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTION_SIZE_HPP

//==--- ripple/core/execution/thread_index.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  thread_index.hpp
/// \brief This file defines thread indexing functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_THREAD_INDEX_HPP
#define RIPPLE_EXECUTION_THREAD_INDEX_HPP

#include "detail/thread_index_impl_.hpp"

namespace ripple {

/*==--- [sizes] ------------------------------------------------------------==*/

/**
 * Gets the value of the block size in the grid in a given dimension. The
 * dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
 * index of the dimension.
 *
 * \param  dim the dimension to get the block size for.
 * \tparam Dim the type of the dimension specifier.
 * \return The size of the block for the dimension (number of threads).
 */
template <typename Dim>
ripple_host_device inline auto block_size(Dim&& dim) noexcept -> size_t {
  return detail::block_size(std::forward<Dim>(dim));
}

/**
 * Gets the value of the grid size in the grid (number of blocks in the
 * grid) in a given dimension. The dimension must be one of dim_x, dim_y,
 * dim_z, or a value specifting the index of the dimension.
 *
 * \note This is the size of the grid local to the device on which the code is
 *       executing. If there are multiple devices being used, each has its own
 *       size.
 *
 * \param  dim the dimension to get the block size for.
 * \tparam Dim the type of the dimension specifier.
 * \return The size of the grid in the given dimension (number of blocks).
 */
template <typename Dim>
ripple_host_device inline auto grid_size(Dim&& dim) noexcept -> size_t {
  return detail::grid_size(std::forward<Dim>(dim));
}

/**
 * Gets the value of the global size of the grid (number of threads in the
 * grid) in a given dimension. The dimension must be one of dim_x, dim_y,
 * dim_z, or a value specifting the index of the dimension.
 *
 * \note This is the size of the grid local to the device on which the code is
 *       executing. If there are multiple devices being used, each has its own
 *       size.
 *
 * \param  dim the dimension to get the block size for.
 * \tparam Dim the type of the dimension specifier.
 * \return The global size of the grid in the given dimension (number of
 *         threads).
 */
template <typename Dim>
ripple_host_device inline auto global_size(Dim&& dim) noexcept -> size_t {
  return detail::grid_size(dim) * detail::block_size(dim);
}

/*==--- [indexing] ---------------------------------------------------------==*/

/**
 * Gets the value of the thread index int he block in the given dimension.
 * The dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
 * index of the dimension.
 *
 * \note This index is local to the execution device, if multiple devices are
 *       used, the same thread index may be on both devices.
 *
 * \param  dim the dimension to get the thread index for.
 * \tparam Dim the type of the dimension specifier.
 * \return The index of the thread in the given dimension for the block its in.
 */
template <typename Dim>
ripple_host_device inline auto thread_idx(Dim&& dim) noexcept -> size_t {
  return detail::thread_idx(std::forward<Dim>(dim));
}

/**
 * Gets the value of the block index in the grid in the given dimension.
 * The dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
 * index of the dimension.
 *
 * \note This index is local to the execution device, if multiple devices are
 *       used, the same block index may be on both devices.
 *
 * \param  dim the dimension to get the index for.
 * \tparam Dim the type of the dimension specifier.
 * \return The index of the thread in the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto block_idx(Dim&& dim) noexcept -> size_t {
  return detail::block_idx(std::forward<Dim>(dim));
}

/**
 * Gets the value of the global index in the grid in the given dimension.
 * The dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
 * index of the dimension.
 *
 * \note This index is local to the execution device, if multiple devices are
 *       used, the same global index may be on both devices.
 *
 * \param  dim the dimension to get the index for.
 * \tparam Dim the type of the dimension specifier.
 * \return The global index of the thread in the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto global_idx(Dim&& dim) noexcept -> size_t {
  return detail::global_idx(std::forward<Dim>(dim));
}

/**
 * Gets the value of the global normalized index in the grid in the given
 * dimension. The dimension must be one of dim_x, dim_y, dim_z, or a value
 * specifting the index of the dimension.
 *
 * \note This index is local to the execution device, if multiple devices are
 *       used, the same global index may be on both devices.
 *
 * \param  dim the dimension to get the index for.
 * \tparam Dim the type of the dimension specifier.
 * \return The global normalized index of the thread in the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto global_norm_idx(Dim&& dim) noexcept -> double {
  return (static_cast<double>(global_idx(dim)) + 0.5) /
         static_cast<double>(detail::global_elements(dim));
}

/**
 * Gets the value of the normalized index in the block in the given
 * dimension. The dimension must be one of dim_x, dim_y, dim_z, or a value
 * specifting the index of the dimension.
 *
 * \param  dim the dimension to get the index for.
 * \tparam Dim the type of the dimension specifier.
 * \return The normalized index of the thread in block for the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto block_norm_idx(Dim&& dim) noexcept -> double {
  return static_cast<double>(block_idx(dim)) /
         static_cast<double>(block_size(dim));
}

/*==--- [utilities] --------------------------------------------------------==*/

/**
 * Determines if a thread is the first thread in its block for the given
 * dimension.
 * \param  dim The dimension to check if this is the first thread of.
 * \tparam Dim The type of the dimension specifier.
 * \return true if the executing thread is the first thread in the block.
 */
template <typename Dim>
ripple_host_device inline auto
first_thread_in_block(Dim&& dim) noexcept -> bool {
  return detail::thread_idx(std::forward<Dim>(dim)) == size_t{0};
}

/**
 * Determines if a thread is the first thread in its block, which requires that
 * its the first thread in each dimension.
 * \return true if the executing thread is the first thread in the block.
 */
ripple_host_device inline auto first_thread_in_block() noexcept -> bool {
  return first_thread_in_block(dim_x) && first_thread_in_block(dim_y) &&
         first_thread_in_block(dim_z);
}

/**
 * Determines if the executing thread is the last thread in the block for the
 * given dimension.
 * \param  dim The dimension to check in.
 * \tparam Dim The type of the dimension specifier.
 * \return true if this is the last thread in the block.
 */
template <typename Dim>
ripple_host_device inline auto
last_thread_in_block(Dim&& dim) noexcept -> bool {
  return detail::thread_idx(std::forward<Dim>(dim)) ==
         (detail::block_size(std::forward<Dim>(dim)) - 1);
}

/**
 * Determines if the executing thread is the last thread in the block, which
 * requries that its the last thread in each dimension, for a specific
 * dimension.
 * \param  dim The dimension to check in.
 * \tparam Dim The type of the dimension specifier.
 *  \return true if this is the last thread in the block.
 */
ripple_host_device inline auto last_thread_in_block() noexcept -> bool {
  return last_thread_in_block(dim_x) && last_thread_in_block(dim_y) &&
         last_thread_in_block(dim_z);
}

/**
 * Determines if this is the first thread in the grid for the given dimension.
 * \param  dim The dimension to check for.
 * \tparam Dim The type of the dimension specifier.
 * \return true if this is the first thread in the grid for the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto
first_thread_in_grid(Dim&& dim) noexcept -> bool {
  return detail::global_idx(std::forward<Dim>(dim)) == size_t{0};
}

/**
 * Determines if the excecuting thread is the first thread in the grid for the
 * device it is executing on.
 * \return true if this is the first thread in the grid.
 */
ripple_host_device inline auto first_thread_in_grid() noexcept -> bool {
  return first_thread_in_grid(dim_x) && first_thread_in_grid(dim_y) &&
         first_thread_in_grid(dim_z);
}

/**
 * Determines if this is the last thread in the grid for the given dimension.
 * \param  dim The dimension to check for.
 * \tparam Dim The type of the dimension specifier.
 * \return true if this is the first thread in the grid for the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto last_thread_in_grid(Dim&& dim) noexcept -> bool {
  return detail::global_idx(std::forward<Dim>(dim)) ==
         (detail::global_elements(std::forward<Dim>(dim)) - 1);
}

/**
 * Determines if this is the last thread in the grid.
 * \return true if this is the first thread in the grid.
 */
ripple_host_device inline auto last_thread_in_grid() noexcept -> bool {
  return last_thread_in_grid(dim_x) && last_thread_in_grid(dim_y) &&
         last_thread_in_grid(dim_z);
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_THREAD_INDEX_HPP

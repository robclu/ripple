//==--- ripple/execution/thread_index.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
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

//==--- [indexing] ---------------------------------------------------------==//

/// Returns the value of the thread index in the block in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
/// index of the dimension.
/// \param  dim the dimension to get the thread index for.
/// \tparam Dim the type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto thread_idx(Dim&& dim) -> std::size_t {
  return detail::thread_idx(std::forward<Dim>(dim));
}

/// Returns the value of the block index in the grid in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
/// index of the dimension.
/// \param  dim the dimension to get the block index for.
/// \tparam Dim the type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto block_idx(Dim&& dim) -> std::size_t {
  return detail::block_idx(std::forward<Dim>(dim));
}

/// Returns the value of the thread index globally in the grid in a given
/// dimension. The dimension must be one of dim_x, dim_y, dim_z, or a value
/// specifting the index of the dimension.
/// \param  dim the dimension to get the global index for.
/// \tparam Dim the type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto global_idx(Dim&& dim) -> std::size_t {
  return detail::global_idx(std::forward<Dim>(dim));
}

//==--- [sizes] ------------------------------------------------------------==//

/// Returns the value of the block size in the grid in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
/// index of the dimension.
/// \param  dim the dimension to get the block size for.
/// \tparam Dim the type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto block_size(Dim&& dim) -> std::size_t {
  return detail::block_size(std::forward<Dim>(dim));
}

/// Returns the value of the grid size in the grid (number of blocks in the
/// grid) in a given dimension. The dimension must be one of dim_x, dim_y,
/// dim_z, or a value specifting the index of the dimension.
/// \param  dim the dimension to get the block size for.
/// \tparam Dim the type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto grid_size(Dim&& dim) -> std::size_t {
  return detail::grid_size(std::forward<Dim>(dim));
}

/// Returns the value of the global size of the grid (number of threads in the
/// grid) in a given dimension. The dimension must be one of dim_x, dim_y,
/// dim_z, or a value specifting the index of the dimension.
/// \param  dim the dimension to get the global size for.
/// \tparam Dim the type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto global_size(Dim&& dim) -> std::size_t {
  return detail::grid_size(dim) * detail::block_size(dim);
}

/// Returns true if the thread is the first thread in the block for dimension
/// \p dim.
/// \param  dim The dimension to check for the first thread.
/// \tparam Dim The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto first_thread_in_block(Dim&& dim) -> bool {
  return detail::thread_idx(dim) == std::size_t{0};
}

/// Returns true if the thread is the last thread in the block for dimension
/// \p dim.
/// \param  dim The dimension to check for the first thread.
/// \tparam Dim The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto last_thread_in_block(Dim&& dim) -> bool {
  return detail::thread_idx(dim) == (detail::block_size(dim) - 1);
}

/// Returns true if the thread is the first thread in the grid for dimension
/// \p dim.
/// \param  dim The dimension to check for the first thread.
/// \tparam Dim The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto first_thread_in_grid(Dim&& dim) -> bool {
  return detail::global_idx(dim) == std::size_t{0};
}

/// Returns true if the thread is the last thread in the block for dimension
/// \p dim.
/// \param  dim The dimension to check for the first thread.
/// \tparam Dim The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto last_thread_in_grid(Dim&& dim) -> bool {
  return detail::global_idx(dim) == (global_size(dim) - 1);
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_THREAD_INDEX_HPP

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

/// Returns the normalized global index in the \p dim dimension, using the
/// global number of threads as the dimension size.
/// \param  dim The dimension to get the normalized index for.
/// \tparam Dim The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto global_norm_idx(Dim&& dim) -> float {
  return static_cast<float>(global_idx(dim)) 
    / static_cast<float>(grid_size(dim));
}

/// Returns the normalized block index in the \p dim dimension, using the number
/// of threads in the block as the block size.
/// \param  dim The dimension to get the normalized index for.
/// \tparam Dim The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto block_norm_idx(Dim&& dim) -> float {
  return static_cast<float>(block_idx(dim)) 
    / static_cast<float>(block_size(dim));
}

/// Returns the normalized global index in the \p dim dimension, using the \p
/// dim_size as the number of elements in the dimension.
///
/// This overload is useful for the case that there may be more threads running
/// than elements in the dimension.
///
/// \param  dim      The dimension to get the normalized index for.
/// \param  dim_size The number of elements in the dimension.
/// \tparam Dim      The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto global_norm_idx(Dim&& dim, std::size_t dim_size)
-> float {
  return static_cast<float>(global_idx(dim)) / static_cast<float>(dim_size);
}

/// Returns the normalized block index in the \p dim dimension, using the \p
/// dim_size as the number of elements in the dimension.
///
/// This overload is useful for the case that some threads in a block do not
/// run.
///
/// \param  dim      The dimension to get the normalized index for.
/// \param  dim_size The number of elements in the dimension.
/// \tparam Dim      The type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto block_norm_idx(Dim&& dim, std::size_t dim_size)
-> float {
  return static_cast<float>(block_idx(dim)) / static_cast<float>(dim_size);
}


//==--- [utilities] --------------------------------------------------------==//

/// Returns true if the thread is the first thread in the block for all
/// dimensions.
ripple_host_device inline auto first_thread_in_block() -> bool {
  return detail::thread_idx(dim_x) == std::size_t{0} &&
         detail::thread_idx(dim_y) == std::size_t{0} &&
         detail::thread_idx(dim_z) == std::size_t{0};
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

/// Returns true if the thread is the first thread in the grid for all
/// dimensions.
ripple_host_device inline auto first_thread_in_grid() -> bool {
  return detail::global_idx(dim_x) == std::size_t{0} &&
         detail::global_idx(dim_y) == std::size_t{0} &&
         detail::global_idx(dim_z) == std::size_t{0};
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

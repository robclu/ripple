//==--- ripple/execution/detail/thread_index_impl_.hpp ----- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  thread_index_impl_.hpp
/// \brief This file implements thread indexing functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_DETAIL_THREAD_INDEX_IMPL__HPP 
#define RIPPLE_EXECUTION_DETAIL_THREAD_INDEX_IMPL__HPP

#include "../execution_traits.hpp"
#include <ripple/utility/dim.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple::detail {

/// The Dim3 class is stores information about the size/index in 3 dimensions.
struct Dim3 {
  std::size_t x = 0;  //!< X value.
  std::size_t y = 0;  //!< Y value.
  std::size_t z = 0;  //!< Z value.
};

/// Values of the thread indices.
static thread_local Dim3 thread_idx_;
/// Values of the block indices.
static thread_local Dim3 block_idx_;
/// Sizes of the blocks.
static thread_local Dim3 block_dim_;
/// Sizes of the grid.
static thread_local Dim3 grid_dim_;

#if defined(__CUDA__) && defined(__CUDA_ARCH__)

//==--- [thread idx] -------------------------------------------------------==//

/// Returns the index of the thread in the block for the x dimension.
ripple_host_device inline auto thread_idx(dimx_t) -> std::size_t {
  return threadIdx.x;
}

/// Returns the index of the thread in the block for the y dimension.
ripple_host_device inline auto thread_idx(dimy_t) -> std::size_t {
  return threadIdx.y;
}

/// Returns the index of the thread in the thread for the z dimension.
ripple_host_device inline auto thread_idx(dimz_t) -> std::size_t {
  return threadIdx.z;
}

/// Returns the index of the thread in the block in the \p dim dimension.
/// \param dim The dimension to get thread index for.
ripple_host_device inline auto thread_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? thread_idx(dim_x) :
         dim == dimy_t::value ? thread_idx(dim_y) :
         dim == dimz_t::value ? thread_idx(dim_z) : 0;
}

//==--- [block idx] -------------------------------------------------------==//

/// Returns the index of the block in the grid for the x dimension.
ripple_host_device inline auto block_idx(dimx_t) -> std::size_t {
  return blockIdx.x;
}

/// Returns the index of the block in the grid for the y dimension.
ripple_host_device inline auto block_idx(dimy_t) -> std::size_t {
  return blockIdx.y;
}

/// Returns the index of the block in the grid for the z dimension.
ripple_host_device inline auto block_idx(dimz_t) -> std::size_t {
  return blockIdx.z;
}

/// Returns the index of the block in the grid in the \p dim dimension.
/// \param dim The dimension to get block index for.
ripple_host_device inline auto block_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? block_idx(dim_x) :
         dim == dimy_t::value ? block_idx(dim_y) :
         dim == dimz_t::value ? block_idx(dim_z) : 0;
}

//==--- [global idx] -------------------------------------------------------==//

/// Returns the index of the thread in the grid for the x dimension.
ripple_host_device inline auto global_idx(dimx_t) -> std::size_t {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

/// Returns the index of the thread in the grid for the y dimension.
ripple_host_device inline auto global_idx(dimy_t) -> std::size_t {
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/// Returns the index of the thread in the grid for the z dimension.
ripple_host_device inline auto global_idx(dimz_t) -> std::size_t {
  return threadIdx.z + blockIdx.z * blockDim.z;
}

/// Returns the index of the thread in the grid in the \p dim dimension.
/// \param dim The dimension to get grid index for.
ripple_host_device inline auto global_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? global_idx(dim_x) :
         dim == dimy_t::value ? global_idx(dim_y) :
         dim == dimz_t::value ? global_idx(dim_z) : 0;
}

//==--- [block size] -------------------------------------------------------==//

/// Returns the size of the block in the grid for the x dimension.
ripple_host_device inline auto block_size(dimx_t) -> std::size_t {
  return blockDim.x;
}

/// Returns the size of the block in the grid for the y dimension.
ripple_host_device inline auto block_size(dimy_t) -> std::size_t {
  return blockDim.y;
}

/// Returns the size of the block in the grid for the z dimension.
ripple_host_device inline auto block_size(dimz_t) -> std::size_t {
  return blockDim.z;
}

/// Returns the size of the block in the grid in the \p dim dimension.
/// \param dim The dimension to get block size for.
ripple_host_device inline auto block_size(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? block_size(dim_x) :
         dim == dimy_t::value ? block_size(dim_y) :
         dim == dimz_t::value ? block_size(dim_z) : 0;
}

//==--- [grid size] -------------------------------------------------------==//

/// Returns the number of blocks in the grid in the x dimension.
ripple_host_device inline auto grid_size(dimx_t) -> std::size_t {
  return gridDim.x;
}

/// Returns the number of blocks in the grid in the y dimension.
ripple_host_device inline auto grid_size(dimy_t) -> std::size_t {
  return gridDim.y;
}

/// Returns the number of blocks in the grid in the z dimension.
ripple_host_device inline auto grid_size(dimz_t) -> std::size_t {
  return gridDim.z;
}

/// Returns the number of blocks in the grid in the \p dim dimension.
/// \param dim The dimension to get grid size for.
ripple_host_device inline auto grid_size(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? grid_size(dim_x) :
         dim == dimy_t::value ? grid_size(dim_y) :
         dim == dimz_t::value ? grid_size(dim_z) : 0;
}

#elif !defined(__CUDA__) || !defined(__CUDA_ARCH__) // __CUDA_ARCH__

//==--- [thread idx] -------------------------------------------------------==//

/// Returns the index of the thread in the block for the x dimension.
ripple_host_device inline auto thread_idx(dimx_t) -> std::size_t {
  return thread_idx_.x;
}

/// Returns the index of the thread in the block for the y dimension.
ripple_host_device inline auto thread_idx(dimy_t) -> std::size_t {
  return thread_idx_.y;
}

/// Returns the index of the thread in the block for the z dimension.
ripple_host_device inline auto thread_idx(dimz_t) -> std::size_t {
  return thread_idx_.z;
}

/// Returns the index of the thread in the block in the \p dim dimension.
/// \param dim The dimension to get grid index for.
ripple_host_device inline auto thread_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? thread_idx(dim_x) :
         dim == dimy_t::value ? thread_idx(dim_y) :
         dim == dimz_t::value ? thread_idx(dim_z) : 0;
}

//==--- [block idx] -------------------------------------------------------==//

/// Returns the index of the block in the grid for the x dimension.
ripple_host_device inline auto block_idx(dimx_t) -> std::size_t {
  return block_idx_.x;
}

/// Returns the index of the block in the grid for the y dimension.
ripple_host_device inline auto block_idx(dimy_t) -> std::size_t {
  return block_idx_.y;
}

/// Returns the index of the block in the grid for the z dimension.
ripple_host_device inline auto block_idx(dimz_t) -> std::size_t {
  return block_idx_.z;
}

/// Returns the index of the block in the grid in the \p dim dimension.
/// \param dim The dimension to get block index for.
ripple_host_device inline auto block_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? block_idx(dim_x) :
         dim == dimy_t::value ? block_idx(dim_y) :
         dim == dimz_t::value ? block_idx(dim_z) : 0;
}

//==--- [grid idx] ---------------------------------------------------------==//

/// Returns the index of the thread in the grid for the x dimension.
ripple_host_device inline auto global_idx(dimx_t) -> std::size_t {
  return block_idx_.x * block_dim_.x + thread_idx_.x;
}

/// Returns the index of the thread in the grid for the y dimension.
ripple_host_device inline auto global_idx(dimy_t) -> std::size_t {
  return block_idx_.y * block_dim_.y + thread_idx_.y;
}

/// Returns the index of the thread in the grid for the z dimension.
ripple_host_device inline auto global_idx(dimz_t) -> std::size_t {
  return block_idx_.z * block_dim_.z + thread_idx_.z;
}

/// Returns the index of the thread in the grid in the \p dim dimension.
/// \param dim The dimension to get grid index for.
ripple_host_device inline auto global_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? global_idx(dim_x) :
         dim == dimy_t::value ? global_idx(dim_y) :
         dim == dimz_t::value ? global_idx(dim_z) : 0;
}

//==--- [block size] -------------------------------------------------------==//

/// Returns the size of the block in the grid for the x dimension.
ripple_host_device inline auto block_size(dimx_t) -> std::size_t {
  return block_dim_.x;
}

/// Returns the size of the block in the grid for the y dimension.
ripple_host_device inline auto block_size(dimy_t) -> std::size_t {
  return block_dim_.y;
}

/// Returns the size of the block in the grid for the z dimension.
ripple_host_device inline auto block_size(dimz_t) -> std::size_t {
  return block_dim_.z;
}

/// Returns the size of the block in the grid in the \p dim dimension.
/// \param dim The dimension to get block size for.
ripple_host_device inline auto block_size(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? block_size(dim_x) :
         dim == dimy_t::value ? block_size(dim_y) :
         dim == dimz_t::value ? block_size(dim_z) : 0;
}

//==--- [grid size] -------------------------------------------------------==//

/// Returns the number of blocks in the grid in the x dimension.
ripple_host_device inline auto grid_size(dimx_t) -> std::size_t {
  return grid_dim_.x;
}

/// Returns the number of blocks in the grid in the y dimension.
ripple_host_device inline auto grid_size(dimy_t) -> std::size_t {
  return grid_dim_.y;
}

/// Returns the number of blocks in the grid in the z dimension.
ripple_host_device inline auto grid_size(dimz_t) -> std::size_t {
  return grid_dim_.z;
}

/// Returns the number of blocks in the grid in the \p dim dimension.
/// \param dim The dimension to get grid size for.
ripple_host_device inline auto grid_size(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? grid_size(dim_x) :
         dim == dimy_t::value ? grid_size(dim_y) :
         dim == dimz_t::value ? grid_size(dim_z) : 0;
}

#endif // __CUDACC__

} // namespace ripple::detail

#endif // RIPPLE_EXECUTION_DETAIL_THREAD_INDEX_IMPL_HPP

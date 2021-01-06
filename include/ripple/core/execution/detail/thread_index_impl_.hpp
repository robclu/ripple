//==--- .../core/execution/detail/thread_index_impl_.hpp --- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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
#include "../../utility/dim.hpp"

namespace ripple::detail {

/**
 * The Dim3 class is stores information about the size/index in 3 dimensions.
 * This is a wrapper class to prvide the same functionality as cuda's dim3.
 */
struct Dim3 {
  size_t x = 0; //!< x value.
  size_t y = 0; //!< y value.
  size_t z = 0; //!< z value.
};

// clang-format off
/**  Values of the thread indices. */
static thread_local Dim3   thread_idx_;
/** Values of the block indices. */
static thread_local Dim3   block_idx_;
/** Sizes of the blocks. */
static thread_local Dim3   block_dim_;
/** Sizes of the grid. */
static thread_local Dim3   grid_dim_;
/** Number of elements in the grid. */
static thread_local Dim3   grid_elements_;
// clang-format on

/** The number of elements in each dimension. */
ripple_device size_t global_elements_for_device_[3];

#if (defined(__CUDA__) && defined(__CUDA_ARCH__)) || \
  (defined(__CUDACC__) && defined(__CUDA_ARCH__))

/**
 * Gets the number of elements in the dimension, globally, for a single
 * device. This can be used for the case that there are multiple gpus to
 * determine exactly how many elements are in each dimension for the gpu.
 *
 * \param  dim The dimension to get the number of elements for.
 * \tparam Dim The type of the dimension specifier.
 * \return The number of elements for a single gpu for the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto global_elements(Dim&& dim) noexcept -> size_t& {
  return global_elements_for_device_[dim];
}

/*==--- [thread idx] -------------------------------------------------------==*/

/**
 * \return The index of the thread in the block for the x dimension.
 */
ripple_host_device inline auto thread_idx(DimX) noexcept -> size_t {
  return threadIdx.x;
}

/**
 * \return The index of the thread in the block for the y dimension.
 */
ripple_host_device inline auto thread_idx(DimY) noexcept -> size_t {
  return threadIdx.y;
}

/**
 * \return The index of the thread in the block for the z dimension.
 */
ripple_host_device inline auto thread_idx(DimZ) noexcept -> size_t {
  return threadIdx.z;
}

/**
 * Gets the index of the thread in the block in the given dimension.
 * \param dim The dimension to get thread index for.
 * \return The index of the the thread in the block for the given dimension.
 */
ripple_host_device inline auto thread_idx(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? thread_idx(dimx())
         : dim == DimY::value ? thread_idx(dimy())
         : dim == DimZ::value ? thread_idx(dimz())
                              : 0;
}

/*==--- [block idx] --------------------------------------------------------==*/

/**
 * \return the index of the block in the grid for the x dimension.
 */
ripple_host_device inline auto block_idx(DimX) noexcept -> size_t {
  return blockIdx.x;
}

/**
 * \return The index of the block in the grid for the y dimension.
 */
ripple_host_device inline auto block_idx(DimY) noexcept -> size_t {
  return blockIdx.y;
}

/**
 * \return The index of the block in the grid for the z dimension.
 */
ripple_host_device inline auto block_idx(DimZ) noexcept -> size_t {
  return blockIdx.z;
}

/**
 * Gets the index of the block in the grid in the given dimension.
 * \param dim The dimension to get block index for.
 * \return The index of the block in the grid in the given dimension.
 */
ripple_host_device inline auto block_idx(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? block_idx(dimx())
         : dim == DimY::value ? block_idx(dimy())
         : dim == DimZ::value ? block_idx(dimz())
                              : 0;
}

/*==--- [global idx] -------------------------------------------------------==*/

/**
 * \return The index of the thread in the grid for the x dimension.
 */

ripple_host_device inline auto global_idx(DimX) noexcept -> size_t {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

/**
 * \return The index of the thread in the grid for the y dimension.
 */
ripple_host_device inline auto global_idx(DimY) noexcept -> size_t {
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/**
 * \return The index of the thread in the grid for the z dimension.
 */
ripple_host_device inline auto global_idx(DimZ) noexcept -> size_t {
  return threadIdx.z + blockIdx.z * blockDim.z;
}

/**
 * Gets the index of the thread in the grid in the given dimension.
 * \param dim The dimension to get grid index for.
 */
ripple_host_device inline auto global_idx(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? global_idx(DimX())
         : dim == DimY::value ? global_idx(DimY())
         : dim == DimZ::value ? global_idx(DimZ())
                              : 0;
}

/*==--- [block size] -------------------------------------------------------==*/

/**
 * \return The size of the block in the grid for the x dimension.
 */
ripple_host_device inline auto block_size(DimX) noexcept -> size_t {
  return blockDim.x;
}

/**
 * \return The size of the block in the grid for the y dimension.
 */
ripple_host_device inline auto block_size(DimY) noexcept -> size_t {
  return blockDim.y;
}

/**
 * \return The size of the block in the grid for the z dimension.
 */
ripple_host_device inline auto block_size(DimZ) noexcept -> size_t {
  return blockDim.z;
}

/**
 * Gets the size of the block in the grid in the dim dimension.
 * \param dim The dimension to get block size for.
 * \return The size of the block in the grid in the given dimension.
 */
ripple_host_device inline auto block_size(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? block_size(dimx())
         : dim == DimY::value ? block_size(dimy())
         : dim == DimZ::value ? block_size(dimz())
                              : 0;
}

/*==--- [grid size] --------------------------------------------------------==*/

/**
 * \return The number of blocks in the grid in the x dimension.
 */
ripple_host_device inline auto grid_size(DimX) noexcept -> size_t {
  return gridDim.x;
}

/**
 * \return The number of blocks in the grid in the y dimension.
 */
ripple_host_device inline auto grid_size(DimY) noexcept -> size_t {
  return gridDim.y;
}

/**
 * \return The number of blocks in the grid in the z dimension.
 */
ripple_host_device inline auto grid_size(DimZ) noexcept -> size_t {
  return gridDim.z;
}

/**
 * Gets the number of blocks in the grid in the given dimension.
 * \param dim The dimension to get grid size for.
 * \return The number of blocks in the grid in the given dimension.
 */
ripple_host_device inline auto grid_size(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? grid_size(dimx())
         : dim == DimY::value ? grid_size(dimy())
         : dim == DimZ::value ? grid_size(dimz())
                              : 0;
}

#elif !defined(__CUDA__) || !defined(__CUDA_ARCH__) // __CUDA_ARCH__

/*==--- [thread idx] -------------------------------------------------------==*/

/**
 * \return The index of the thread in the block for the x dimension.
 */
ripple_host_device inline auto thread_idx(DimX) noexcept -> size_t {
  return thread_idx_.x;
}

/**
 * \return The index of the thread in the block for the y dimension.
 */
ripple_host_device inline auto thread_idx(DimY) noexcept -> size_t {
  return thread_idx_.y;
}

/**
 * \return The index of the thread in the block for the z dimension.
 */
ripple_host_device inline auto thread_idx(DimZ) noexcept -> size_t {
  return thread_idx_.z;
}

/**
 * Gets the index of the thread in the block in the given dimension.
 * \param dim The dimension to get grid index for.
 * \return The index of the thread in the block in the given dimension.
 */
ripple_host_device inline auto thread_idx(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? thread_idx(dimx())
         : dim == DimY::value ? thread_idx(dimy())
         : dim == DimZ::value ? thread_idx(dimz())
                              : 0;
}

/*==--- [block idx] -------------------------------------------------------==*/

/**
 * \return The index of the block in the grid for the x dimension.
 */
ripple_host_device inline auto block_idx(DimX) noexcept -> size_t {
  return block_idx_.x;
}

/**
 * \return The index of the block in the grid for the y dimension.
 */
ripple_host_device inline auto block_idx(DimY) noexcept -> size_t {
  return block_idx_.y;
}

/**
 * \return The index of the block in the grid for the z dimension.
 */
ripple_host_device inline auto block_idx(DimZ) noexcept -> size_t {
  return block_idx_.z;
}

/**
 * Gets the index of the block in the grid in the \p dim dimension.
 * \param dim The dimension to get block index for.
 * \return The index of the block in the grid in the given dimension.
 */
ripple_host_device inline auto block_idx(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? block_idx(dimx())
         : dim == DimY::value ? block_idx(dimy())
         : dim == DimZ::value ? block_idx(dimz())
                              : 0;
}

/*==--- [grid idx] ---------------------------------------------------------==*/

/**
 * \return The index of the thread in the grid for the x dimension.
 */
ripple_host_device inline auto global_idx(DimX) noexcept -> size_t {
  return block_idx_.x * block_dim_.x + thread_idx_.x;
}

/**
 * \return The index of the thread in the grid for the y dimension.
 */
ripple_host_device inline auto global_idx(DimY) noexcept -> size_t {
  return block_idx_.y * block_dim_.y + thread_idx_.y;
}

/**
 * \return The index of the thread in the grid for the z dimension.
 */
ripple_host_device inline auto global_idx(DimZ) noexcept -> size_t {
  return block_idx_.z * block_dim_.z + thread_idx_.z;
}

/**
 * Gets the index of the thread in the grid in the given dimension.
 * \param dim The dimension to get grid index for.
 * \return The index of the thread in the grid in the given dimension.
 */
ripple_host_device inline auto global_idx(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? global_idx(dimx())
         : dim == DimY::value ? global_idx(dimy())
         : dim == DimZ::value ? global_idx(dimz())
                              : 0;
}

/*==--- [block size] -------------------------------------------------------==*/

/**
 * \return The size of the block in the grid for the x dimension.
 */
ripple_host_device inline auto block_size(DimX) noexcept -> size_t {
  return block_dim_.x;
}

/**
 * \return The size of the block in the grid for the y dimension.
 */
ripple_host_device inline auto block_size(DimY) noexcept -> size_t {
  return block_dim_.y;
}

/**
 * \return The size of the block in the grid for the z dimension.
 */
ripple_host_device inline auto block_size(DimZ) noexcept -> size_t {
  return block_dim_.z;
}

/**
 * Gets the size of the block in the grid in the given dimension.
 * \param dim The dimension to get block size for.
 * \return The size of the block in the grid in the given dimension.
 */
ripple_host_device inline auto block_size(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? block_size(dimx())
         : dim == DimY::value ? block_size(dimy())
         : dim == DimZ::value ? block_size(dimz())
                              : 0;
}

/*==--- [grid size] --------------------------------------------------------==*/

/**
 * \return The number of blocks in the grid in the x dimension.
 */
ripple_host_device inline auto grid_size(DimX) noexcept -> size_t {
  return grid_dim_.x;
}

/**
 * \return The number of blocks in the grid in the y dimension.
 */
ripple_host_device inline auto grid_size(DimY) noexcept -> size_t {
  return grid_dim_.y;
}

/**
 * \return The number of blocks in the grid in the z dimension.
 */
ripple_host_device inline auto grid_size(DimZ) noexcept -> size_t {
  return grid_dim_.z;
}

/**
 * Gets the number of blocks in the grid in the given dimension.
 * \param dim The dimension to get grid size for.
 * \return The number of blocks in the grid in the given dimension.
 */
ripple_host_device inline auto grid_size(size_t dim) noexcept -> size_t {
  return dim == DimX::value   ? grid_size(dimx())
         : dim == DimY::value ? grid_size(dimy())
         : dim == DimZ::value ? grid_size(dimz())
                              : 0;
}

/**
 * Gets the number of elements in the dimension, globally, for a single
 * node. This can be used for the case that there are multiple nodes to
 * determine exactly how many elements are assigned to the specific node.
 *
 * \param  dim The dimension to get the number of elements for.
 * \tparam Dim The type of the dimension specifier.
 * \return The number of elements for a single node in the given dimension.
 */
template <typename Dim>
ripple_host_device inline auto global_elements(Dim&& dim) noexcept -> size_t& {
  return dim == DimX::value   ? grid_elements_.x
         : dim == DimY::value ? grid_elements_.y
                              : grid_elements_.z;
}

#endif // __CUDACC__

} // namespace ripple::detail

#endif // RIPPLE_EXECUTION_DETAIL_THREAD_INDEX_IMPL_HPP

//==--- ripple/multidim/detail/thread_index_impl_.hpp ------ -*- C++ -*- ---==//
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

#ifndef RIPPLE_MULTIDIM_DETAIL_THREAD_INDEX_IMPL__HPP
#define RIPPLE_MULTIDIM_DETAIL_THREAD_INDEX_IMPL__HPP

#include <ripple/utility/dim.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple::detail {

/// Value of the thread index in the x dimension.
static thread_local std::size_t thread_idx_x = 0;
/// Value of the thread index in the y dimension,
static thread_local std::size_t thread_idx_y = 0;
/// Value of the thread index in the z dimension.
static thread_local std::size_t thread_idx_z = 0;

#if defined(__CUDA__) && defined(__CUDA_ARCH__)

//==--- [flattened idx] ----------------------------------------------------==//

/// Returns the index of the thread in the grid for the x dimension.
ripple_host_device inline auto grid_idx(dimx_t) -> std::size_t {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

/// Returns the index of the thread in the grid for the y dimension.
ripple_host_device inline auto grid_idx(dimy_t) -> std::size_t {
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/// Returns the index of the thread in the grid for the z dimension.
ripple_host_device inline auto grid_idx(dimz_t) -> std::size_t {
  return threadIdx.z + blockIdx.z * blockDim.z;
}

/// Returns the index of the thread in the grid in the \p dim dimension.
/// \param dim The dimension to get grid index for.
ripple_host_device inline auto grid_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? grid_idx(dim_x) :
         dim == dimy_t::value ? grid_idx(dim_y) :
         dim == dimz_t::value ? grid_idx(dim_z) : 0;
}

/// Returns the index of the thread in the grid for the x dimension, with \p
/// params for the execution space.
/// \param  params The execution parameters for the grid space.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(dimx_t, const Params& params)
-> std::size_t { 
  using params_t = std::decay_t<Params>;
  return blockIdx.x * blockDim.x * params_t::grain
    + threadIdx.x + params.grain_index * blockDim.x;
}

/// Returns the index of the thread in the grid for the y dimension.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(dimy_t, Params&&)
-> std::size_t {
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/// Returns the index of the thread in the grid for the z dimension.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(dimz_t, Params&&)
-> std::size_t {
  return threadIdx.z + blockIdx.z * blockDim.z;
}

/// Returns the index of the thread in the grid for the \p dim dimension.
/// \param  dim    The dimension to get the grid index for.
/// \param  params The parameters which define the grid execution space.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(std::size_t dim, const Params& params)
-> std::size_t {
  return dim == dimx_t::value ? grid_idx(dim_x, params) :
         dim == dimy_t::value ? grid_idx(dim_y, params) :
         dim == dimz_t::value ? grid_idx(dim_z, params) : 0;
}

#elif !defined(__CUDA__) || !defined(__CUDA_ARCH__) // __CUDA_ARCH__

/// Returns the index of the thread in the grid for the x dimension.
ripple_host_device inline auto grid_idx(dimx_t) -> std::size_t {
  return thread_idx_x;
}

/// Returns the index of the thread in the grid for the y dimension.
ripple_host_device inline auto grid_idx(dimy_t) -> std::size_t {
  return thread_idx_y;
}

/// Returns the index of the thread in the grid for the z dimension.
ripple_host_device inline auto grid_idx(dimz_t) -> std::size_t {
  return thread_idx_z;
}

/// Returns the index of the thread in the grid in the \p dim dimension.
/// \param dim The dimension to get grid index for.
ripple_host_device inline auto grid_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? grid_idx(dim_x) :
         dim == dimy_t::value ? grid_idx(dim_y) :
         dim == dimz_t::value ? grid_idx(dim_z) : 0;
}

/// Returns the index of the thread in the grid for the x dimension, with \p
/// params for the execution space.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(dimx_t, Params&&)
-> std::size_t {
  return thread_idx_x;
}

/// Returns the index of the thread in the grid for the y dimension.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(dimy_t, Params&& params)
-> std::size_t {
  return thread_idx_y;
}

/// Returns the index of the thread in the grid for the z dimension.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(dimz_t, Params&& params)
-> std::size_t {
  return thread_idx_z;
}

/// Returns the index of the thread in the grid for the \p dim dimension.
/// \param  dim    The dimension to get the grid index for.
/// \param  params The parameters which define the grid execution space.
/// \tparam Params The type of the execution parameters.
template <typename Params>
ripple_host_device inline auto grid_idx(std::size_t dim, Params&& params)
-> std::size_t {
  return dim == dimx_t::value ? grid_idx(dim_x, params) :
         dim == dimy_t::value ? grid_idx(dim_y, params) :
         dim == dimz_t::value ? grid_idx(dim_z, params) : 0;
}

#endif // __CUDACC__

} // namespace ripple::detail

#endif // RIPPLE_MULTIDIM_DETAIL_THREAD_INDEX_IMPL_HPP

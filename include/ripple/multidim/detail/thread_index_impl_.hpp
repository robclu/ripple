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

/// Returns the index of the thread if the dimensional space was flattened in
/// the x direction.
ripple_host_device inline auto flattened_idx(dimx_t) -> std::size_t {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the y direction.
ripple_host_device inline auto flattened_idx(dimy_t) -> std::size_t {
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the z direction.
ripple_host_device inline auto flattened_idx(dimz_t) -> std::size_t {
  return threadIdx.z + blockIdx.z * blockDim.z;
}

/// Returns the flattened index in the \p dim dimension.
/// \param dim The dimension to get the flattened index for.
ripple_host_device inline auto flattened_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? flattened_idx(dim_x) :
         dim == dimy_t::value ? flattened_idx(dim_y) :
         dim == dimz_t::value ? flattened_idx(dim_z) : 0;
}

#elif defined(__CUDA__) && !defined(__CUDA_ARCH__) // __CUDA_ARCH__

/// Returns the index of the thread if the dimensional space was flattened in
/// the x direction.
ripple_host_device inline auto flattened_idx(dimx_t) -> std::size_t {
  return thread_idx_x;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the y direction.
ripple_host_device inline auto flattened_idx(dimy_t) -> std::size_t {
  return thread_idx_y;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the z direction.
ripple_host_device inline auto flattened_idx(dimz_t) -> std::size_t {
  return thread_idx_z;
}

/// Returns the flattened index in the \p dim dimension.
/// \param dim The dimension to get the flattened index for.
ripple_host_device inline auto flattened_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? flattened_idx(dim_x) :
         dim == dimy_t::value ? flattened_idx(dim_y) :
         dim == dimz_t::value ? flattened_idx(dim_z) : 0;
}

#endif // __CUDACC__

} // namespace ripple

#endif // RIPPLE_MULTIDIM_DETAIL_THREAD_INDEX_IMPL_HPP

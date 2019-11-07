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

#if defined(__CUDACC__)

//==--- [flattened idx] ----------------------------------------------------==//

/// Returns the index of the thread if the dimensional space was flattened in
/// the x direction.
ripple_device_only inline auto flattened_idx(dimx_t) -> std::size_t {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the y direction.
ripple_device_only inline auto flattened_idx(dimy_t) -> std::size_t {
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/// Returns the index of the thread if the dimensional space was flattened in
/// the z direction.
ripple_device_only inline auto flattened_idx(dimz_t) -> std::size_t {
  return threadIdx.z + blockIdx.z * blockDim.z;
}

/// Returns the flattened index in the \p dim dimension.
/// \param dim The dimension to get the flattened index for.
ripple_device_only inline auto flattened_idx(std::size_t dim) -> std::size_t {
  return dim == dimx_t::value ? flattened_idx(dim_x) :
         dim == dimy_t::value ? flattened_idx(dim_y) :
         dim == dimz_t::value ? flattened_idx(dim_z) : 0;
}

#endif // __CUDACC__

} // namespace ripple

#endif // RIPPLE_MULTIDIM_DETAIL_THREAD_INDEX_IMPL_HPP

//==--- fluidity/utility/detail/copy_impl_.cuh ------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  copy_impl_.hpp
/// \brief This file defines a cuda kernel to copy data.
//
//==------------------------------------------------------------------------==//

#include "../portability.hpp"
#include <ripple/multidim/thread_index.hpp>

#ifndef RIPPLE_UTILITY_DETAIL_COPY_IMPL__CUH
#define RIPPLE_UTILITY_DETAIL_COPY_IMPL__CUH

namespace ripple::cuda::kernel {

#if defined(__CUDACC__)

/// Copies each data element from \p in to \p out.
/// \param  in       A pointer to the input data.
/// \param  out      A pointer to the output data.
/// \param  elements The number of elements to set.
/// \tparam Ptr      The type of the pointers.
template <typename Ptr>
ripple_global auto copy(Ptr* out, const Ptr* in, std::size_t elements) -> void {
  const auto idx = grid_idx(dim_x);
  if (idx < elements) {
    out[idx] = in[idx];
  }
}

#endif // __CUDACC__

} // namespace ripple::cuda::kernel

#endif // RIPPLE_UTILITY_DETAIL_COPY_IMPL__CUH



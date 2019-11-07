//==--- ripple/multidim/thread_index.hpp ------------------- -*- C++ -*- ---==//
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

#ifndef RIPPLE_MULTIDIM_THREAD_INDEX_HPP
#define RIPPLE_MULTIDIM_THREAD_INDEX_HPP

#include "detail/thread_index_impl_.hpp"

namespace ripple {

#if defined(__CUDACC__)

/// Returns the value of the flattened thread index in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or else a compile time error
/// will be generated. This returns the global flattened index -- i.e, as if the
/// data were laid out in a single dimension, one dimension after the other.
/// \param  dim The dimension to get the thread index for.
/// \tparam Dim The type of the dimension specifier.
template <typename Dim>
ripple_device_only inline auto flattened_idx(Dim&& dim) -> std::size_t {
  return detail::flattened_idx(std::forward<Dim>(dim));
}

#endif // __CUDACC__

} // namespace ripple

#endif // RIPPLE_MULTIDIM_THREAD_INDEX_HPP

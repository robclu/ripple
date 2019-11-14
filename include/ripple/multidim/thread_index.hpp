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

/// Returns the value of the thread index in the grid in a given dimension. The
/// dimension must be one of dim_x, dim_y, dim_z, or a value specifting the
/// index of the dimension.
/// \param  dim the dimension to get the thread index for.
/// \tparam Dim the type of the dimension specifier.
template <typename Dim>
ripple_host_device inline auto grid_idx(Dim&& dim) -> std::size_t {
  return detail::grid_idx(std::forward<Dim>(dim));
}

/// Returns the value of the thread index in the grid for the dimension \p dim,
/// where the execution space of the grid is defined by the execution \p params.
/// \param  dim    The dimension to get the thread index for.
/// \param  params Parameters which define the execution space.
/// \tparam Dim    The type of the dimension specifier.
/// \tparam Params The type of the execution parameters.
template <typename Dim, typename Params>
ripple_host_device inline auto grid_idx(Dim&& dim, Params&& params)
-> std::size_t {
  return detail::grid_idx(std::forward<Dim>(dim), std::forward<Params>(params));
}

} // namespace ripple

#endif // RIPPLE_MULTIDIM_THREAD_INDEX_HPP

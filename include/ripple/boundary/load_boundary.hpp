//==--- ripple/boundary/load_boundary.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  load_boundary.hpp
/// \brief This file implements functionality to boundary data for a block,
///        both globally and locally.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_LOAD_BOUNDARY_HPP
#define RIPPLE_BOUNDARY_LOAD_BOUNDARY_HPP

#include "detail/load_global_boundary_impl_.hpp"
#include <ripple/execution/execution_traits.hpp>
#include <ripple/execution/thread_index.hpp>

namespace ripple {

/// Loads the boundary data for the \p block, using the \p loader. The loader
/// must be an implementation of the \p BoundaryLoader interface, or a compile
/// time error will be generated.
/// \param  block  The block to load the boundary data for.
/// \param  loader The loader for the data.
/// \tparam Block  The type of the block.
/// \tparam Loader The loader for the block data.
template <typename Block, typename Loader>
ripple_host_device auto load_global_boundary(Block&& block, Loader&& loader)
-> void {
  invoke(block,
    [] (auto it, auto&& loader) {
      constexpr auto dims = it.dimensions();
      using dim_spec_t    = std::conditional_t<
        dims == 1, dimx_t, std::conditional_t<dims == 2, dimy_t, dimz_t>
      >;

      GhostIndex<dims> indices;
      if (!indices.init_as_global(it)) {
        return;
      }

      // Call loader impl ...
      detail::load_global_boundary(
        dim_spec_t{}, it, indices, std::forward<Loader>(loader)
      );
    },
    std::forward<Loader>(loader)
  );
}



} // namespace ripple

#endif // RIPPLE_BOUNDARY_LOAD_BOUNDARY_HPP

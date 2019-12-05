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
#include "detail/load_internal_boundary_impl_.hpp"
#include <ripple/execution/execution_traits.hpp>
#include <ripple/execution/thread_index.hpp>

#include <ripple/multidim/dynamic_multidim_space.hpp>

namespace ripple {

/// Loads the boundary data for the \p block, using the \p loader. The loader
/// must be an implementation of the \p BoundaryLoader interface, or a compile
/// time error will be generated.
/// \param  block  The block to load the boundary data for.
/// \param  loader The loader for the data.
/// \param  args   Additional arguments for the loader.
/// \tparam Block  The type of the block.
/// \tparam Loader The loader for the block data.
/// \tparam Args   Type type of the loader arguments.
template <typename Block, typename Loader, typename... Args>
ripple_host_device auto load_global_boundary(
  Block&& block, Loader&& loader, Args&&... args
) -> void {
  invoke(block,
    [] (auto it, auto&& loader, auto&&... args) {
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
        dim_spec_t{}, it, indices, loader, args...
      );
    },
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  );
}

/// Loads the boundary data for a block pointed do by \p it_to, using the data
/// pointed to by \p it_from, offsetting into the boundary of \p it_to, then
/// performing the offset into \p it_from.
/// \param  it_from      The iterator to load the boundary data from.
/// \param  it_to        The iterator to load the boundary data into.
/// \param  loader       The loader to use to load the data.
/// \param  args         Additional arguments for the loader.
/// \tparam IteratorFrom The type of the from iterator.
/// \tparam IteratorTo   The type of the to iterator.
/// \tparam Loader       The type of the loader.
/// \tparam Args         The type of the arguments.
template <
  std::size_t Dims        ,
  typename    IteratorFrom,
  typename    IteratorTo  ,
  typename    Loader      ,
  typename... Args
>
ripple_host_device auto load_internal_boundary(
    IteratorFrom&& it_from, IteratorTo&& it_to, Loader&& loader, Args&&... args
) -> void {
  constexpr auto dims = Dims;
  using dim_spec_t    = std::conditional_t<
    dims == 1, dimx_t, std::conditional_t<dims == 2, dimy_t, dimz_t>
  >;
 
  // Create the space defining the block size. We need to do this for blocks at
  // the end of the domain for which all threads in the block may not run. 
  DynamicMultidimSpace<dims> space;
  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim  = d;
    const auto elements = it_from.size(dim) - block_idx(dim) * block_size(dim);
    space[dim] = std::min(it_to.size(dim), elements);
  });

  GhostIndex<dims> indices;
  if (!indices.init_as_block(it_to, space)) {
    return;
  }

  // Perform the load ...
  detail::load_internal_boundary(
    dim_spec_t{}                       , 
    std::forward<IteratorFrom>(it_from),
    std::forward<IteratorTo>(it_to)    ,
    indices                            ,
    std::forward<Loader>(loader)       ,
    std::forward<Args>(args)...
  );
}


} // namespace ripple

#endif // RIPPLE_BOUNDARY_LOAD_BOUNDARY_HPP

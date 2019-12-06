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
///
/// Loads boundary data from the \p it_from iterator into the boundary of the \p
/// it_to iterator.
///
/// \pre The iterators must be offset to the locations from which the padding
/// will be loaded. I.e, The top left cell pointed to by an iterator must point
/// to the first cell in the domain.
///
/// If the \p it_from is a smaller (in any dimension) iterator than the \p it_to
/// iterator, the behaviour is undefined. Additionally, it must be possible to
/// offset both itertators by the padding amount of \p it_to in each dimension.
///
/// \param  it_from      The iterator to load the boundary data from.
/// \param  it_to        The iterator to load the boundary data into.
/// \tparam IteratorFrom The type of the from iterator.
/// \tparam IteratorTo   The type of the to iterator.
template <std::size_t Dims, typename IteratorFrom, typename IteratorTo>
ripple_host_device auto load_internal_boundary(
    IteratorFrom&& it_from, IteratorTo&& it_to
) -> void {
  constexpr auto dims = Dims;

  /// More both iterators to the top left of the domain:
  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = d;
    it_from.shift(dim, -static_cast<int>(it_to.padding()));
    it_to.shift(dim, -static_cast<int>(it_to.padding()));
  });

  // Now load in the data by shifting all the threads around the domain;
  detail::load_internal<dims>(it_from, it_to);

  // Shift the iterators back:
  unrolled_for<dims>([&] (auto d) {
    constexpr auto dim = d;
    it_from.shift(dim, static_cast<int>(it_to.padding()));
    it_to.shift(dim, static_cast<int>(it_to.padding()));
  });
}

} // namespace ripple

#endif // RIPPLE_BOUNDARY_LOAD_BOUNDARY_HPP

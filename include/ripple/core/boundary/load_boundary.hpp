//==--- ripple/core/boundary/load_boundary.hpp ------------------ -*- C++ -*-
//---==//
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
#include "boundary_loader.hpp"
#include <ripple/core/execution/execution_traits.hpp>
#include <ripple/core/execution/thread_index.hpp>

namespace ripple {

/**
 * Loads the global boundary data for the \p it iterator using the \p loader.
 *
 * \note The loader must implement the BoundaryLoader interface.
 *
 * \param  it       The iterator to load the boundaries for.
 * \param  loader   The loader which defines how to load the boundaries.
 * \param  args     Additional arguments for the loading.
 * \tparam Iterator The type of the iterator.
 * \tparam Loader   The type of the loader.
 * \tparam Args     The types of the arguments.
 */
template <typename Iterator, typename Loader, typename... Args>
ripple_host_device auto
load_boundary(Iterator&& it, Loader&& loader, Args&&... args) noexcept -> void {
  static_cast(
    is_iterator_v<Iterator>,
    "Boundary loading requires input to be an iterator!");
  static_assert(
    is_loader_v<Loader>,
    "Boundary loading requires a loader which implements the BoundaryLoader "
    "interface!");

  constexpr auto dims = iterator_traits_t<Iterator>::dimensions;
  using DimType       = std::conditional_t<
    dims == 1,
    dimx_t,
    std::conditional_t<dims == 2, dimy_t, dimz_t>>;

  GhostIndex<dims> indices;
  if (!indices.init_as_global(it)) {
    return;
  }

  // Call loader impl ...
  detail::load_global_boundary(
    DimType{},
    static_cast<Iterator&&>(it),
    indices,
    static_cast<Loader&&>(loader),
    static_cast<Args&&>(args)...);
}

/**
 * Loads boundary data from the \p it_from iterator into the boundary of the \p
 * it_to iterator.
 *
 * \pre The iterators must be offset to the locations from which the padding
 *      will be loaded. I.e, The top left cell pointed to by an iterator must *
 *      point to the first cell in the domain.
 *
 * \note If the \p it_from is a smaller (in any dimension) iterator than the \p
 *       it_to iterator, the behaviour is undefined. Additionally, it must be
 *       possible to offset both itertators by the padding amount of \p it_to in
 *       each dimension.
 *
 * \todo Maybe change the dims parameter to be inferred from the iterators.
 *
 * \todo This uses a lot of registers, and since it's mostly used for loading of
 *       shared memory data, it needs to be improved.
 *
 * \param  it_from The iterator to load the boundary data from.
 * \param  it_to   The iterator to load the boundary data into.
 * \tparam Dims    The number of dimension to load data for.
 * \tparam ItFrom  The type of the from iterator.
 * \tparam ItTo    The type of the to iterator.
 */
template <size_t Dims, typename ItFrom, typename ItTo>
ripple_host_device auto
load_internal_boundary(ItFrom&& it_from, ItTo&& it_to) noexcept -> void {
  static_assert(
    is_iterator_v<ItFrom>,
    "Internal boundary loading requires the input to get loading data from to "
    "be an iterator!");
  static_assert(
    is_iterator_v<ItTo>,
    "Internal boundary loading requires the input to get loading data into to "
    "be an iterator!");

  static constexpr size_t dims_from = iterator_traits_t<ItFrom>::dimensions;
  static constexpr size_t dims_to   = iterator_traits_t<ItTo>::dimensions;
  static_assert(
    dims_from >= Dims && dims_to >= Dims,
    "Invalid dimensions for loading of boundary data!");

  // Move both iterators to the top left of the domain:
  unrolled_for<Dims>([&](auto dim) {
    it_from.shift(dim, -static_cast<int>(it_to.padding()));
    it_to.shift(dim, -static_cast<int>(it_to.padding()));
  });

  // Now load in the data by shifting all the threads around the domain;
  detail::load_internal<Dims>(it_from, it_to);

  // Shift the iterators back:
  unrolled_for<Dims>([&](auto dim) {
    it_from.shift(dim, static_cast<int>(it_to.padding()));
    it_to.shift(dim, static_cast<int>(it_to.padding()));
  });
}

} // namespace ripple

#endif // RIPPLE_BOUNDARY_LOAD_BOUNDARY_HPP

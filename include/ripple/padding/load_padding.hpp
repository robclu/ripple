/**=--- ripple/padding/load_padding.hpp -------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  load_padding.hpp
 * \brief This file implements functionality for loading padding data for
 *        iterators over multidimensional spaces, such as blocks and tensors,
 *        both globally and locally.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_LOAD_BOUNDARY_HPP
#define RIPPLE_PADDING_LOAD_BOUNDARY_HPP

#include "detail/load_global_padding_impl_.hpp"
#include "detail/load_internal_padding_impl_.hpp"
#include "padding_loader.hpp"
#include <ripple/execution/execution_traits.hpp>
#include <ripple/execution/thread_index.hpp>

namespace ripple {

/**
 * Loads the global padding data for the iterator using the given loader.
 *
 * \note The loader must implement the PaddingLoader interface.
 *
 * \param  it       The iterator to load the padding for.
 * \param  loader   The loader which defines how to load the padding.
 * \param  args     Additional arguments for the loading.
 * \tparam Iterator The type of the iterator.
 * \tparam Loader   The type of the loader.
 * \tparam Args     The types of the arguments.
 */
template <typename Iterator, typename Loader, typename... Args>
ripple_all auto
load_boundary(Iterator&& it, Loader&& loader, Args&&... args) noexcept -> void {
  static_assert(
    is_iterator_v<Iterator>,
    "Padding loading requires the input to be an iterator!");
  static_assert(
    is_loader_v<Loader>,
    "Padding loading requires a loader which implements the PaddingLoader "
    "interface!");

  constexpr auto   dims = iterator_traits_t<Iterator>::dimensions;
  GhostIndex<dims> indices;
  if (!indices.init_as_global(it)) {
    return;
  }

  // Call loader impl ...
  detail::load_global_padding(
    dim_type_from_dims_t<dims>{},
    ripple_forward(it),
    indices,
    ripple_forward(loader),
    ripple_forward(args)...);
}

/**
 * Functor which applies a loader to call the padding loading function. This
 * can be passed to a graph to be applied to a tensor.
 */
struct LoadPadding {
  /**
   * Loads the global padding data for the iterator using the loader.
   *
   * \note The loader must implement the PaddingLoader interface.
   *
   * \param  it       The iterator to load the padding for.
   * \param  loader   The loader which defines how to load the padding.
   * \param  args     Additional arguments for the loading.
   * \tparam Iterator The type of the iterator.
   * \tparam Loader   The type of the loader.
   * \tparam Args     The types of the arguments.
   */
  template <typename Iterator, typename Loader, typename... Args>
  ripple_all auto
  operator()(Iterator&& it, Loader&& loader, Args&&... args) const noexcept
    -> void {
    static_assert(
      is_iterator_v<Iterator>,
      "Padding loading requires input to be an iterator!");
    static_assert(
      is_loader_v<Loader>,
      "Padding loading requires a loader which implements the PaddingLoader "
      "interface!");

    constexpr auto   dims = iterator_traits_t<Iterator>::dimensions;
    GhostIndex<dims> indices;
    if (!indices.init_as_global(it)) {
      return;
    }

    // Call loader impl ...
    detail::load_global_padding(
      dim_type_from_dims_t<dims>{},
      ripple_forward(it),
      indices,
      ripple_forward(loader),
      ripple_forward(args)...);
  }
};

/**
 * Functor which applies a loader to call the padding loading function. This
 * can be passed to a graph to be applied to a tensor.
 */
struct LoadMultiPadding {
  /**
   * Loads the global padding data for the iterator using the loader.
   *
   * \note The loader must implement the PaddingLoader interface.
   *
   * \param  it       The iterator to load the padding for.
   * \param  loader   The loader which defines how to load the padding.
   * \param  args     Additional arguments for the loading.
   * \tparam Iterator The type of the iterator.
   * \tparam Loader   The type of the loader.
   * \tparam Args     The types of the arguments.
   */
  template <typename ItA, typename ItB, typename Loader, typename... Args>
  ripple_all auto operator()(
    ItA&& it_a, ItB&& it_b, Loader&& loader, Args&&... args) const noexcept
    -> void {
    static_assert(
      is_iterator_v<ItA> && is_iterator_v<ItB>,
      "Padding loading requires inputs to be iterators!");
    static_assert(
      is_loader_v<Loader>,
      "Padding loading requires a loader which implements the PaddingLoader "
      "interface!");

    constexpr auto   dims = iterator_traits_t<ItA>::dimensions;
    GhostIndex<dims> indices;
    if (indices.init_as_global(it_a)) {
      detail::load_global_padding(
        dim_type_from_dims_t<dims>{},
        ripple_forward(it_a),
        indices,
        ripple_forward(loader),
        ripple_forward(args)...);
    }

    constexpr auto dims_b = iterator_traits_t<ItB>::dimensions;
    if (indices.init_as_global(it_b)) {
      detail::load_global_padding(
        dim_type_from_dims_t<dims_b>{},
        ripple_forward(it_b),
        indices,
        ripple_forward(loader),
        ripple_forward(args)...);
    }
  }
};

/**
 * Loads padding data from the from iterator into the padding of the to
 * iterator.
 *
 * \pre The iterators must be offset to the locations from which the padding
 *      will be loaded.
 *
 * \note If the from iterator is a smaller (in any dimension) iterator than the
 *       to iterator, the behaviour is undefined. Additionally, it must be
 *       possible to offset both itertators by the padding amount of to in
 *       each dimension.
 *
 * \todo Maybe change the dims parameter to be inferred from the iterators.
 *
 * \todo This uses a lot of registers, and since it's mostly used for loading of
 *       shared memory data, it needs to be improved.
 *
 * \param  from    The iterator to load the boundary data from.
 * \param  to      The iterator to load the boundary data into.
 * \tparam Dims    The number of dimension to load data for.
 * \tparam ItFrom  The type of the from iterator.
 * \tparam ItTo    The type of the to iterator.
 */
template <size_t Dims, typename ItFrom, typename ItTo>
ripple_all auto
load_internal_boundary(ItFrom&& from, ItTo&& to) noexcept -> void {
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
    from.shift(dim, -static_cast<int>(to.padding()));
    to.shift(dim, -static_cast<int>(to.padding()));
  });

  detail::load_internal<Dims>(from, to);

  // Shift the iterators back:
  unrolled_for<Dims>([&](auto dim) {
    from.shift(dim, static_cast<int>(to.padding()));
    to.shift(dim, static_cast<int>(to.padding()));
  });
}

} // namespace ripple

#endif // RIPPLE_PADDING_LOAD_BOUNDARY_HPP

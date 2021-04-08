/**=--- ripple/padding/detail/load_globaL_padding_impl_.hpp  -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  load_global_padding_impl_.hpp
 * \brief This file implements functionality to load global padding data.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_DETAIL_LOAD_GLOBAL_PADDING_IMPL__HPP
#define RIPPLE_PADDING_DETAIL_LOAD_GLOBAL_PADDING_IMPL__HPP

#include "../ghost_index.hpp"

namespace ripple::detail {

// clang-format off

/**
 * Loads the padding data for the given dimension, using the index in dim
 * from the indices, and the given loader to load the data.
 *
 * \note This implementation uses *a lot* of registers on the gpu, which really
 *       needs to be improved.
 *
 * \param  it       The iterator to load the padding for.
 * \param  indices  The indices for the ghost cells.
 * \param  dim      The dimension to load the cell in.
 * \param  loader   The loader to use to load the cells.
 * \param  args     Additional arguments for the loader.
 * \tparam It       The type of the iterator.
 * \tparam Dims     The number of dimensions for the indices.
 * \tparam Loader   The type of the loader for the dims.
 * \tparam Dim      The dimension to load in.
 * \tparam Args     The type of additional arguments.
 */
template <
  typename It, size_t Dims, typename Loader, typename Dim, typename... Args>
ripple_all auto load_global_padding_for_dim(
  It&&                    it,
  const GhostIndex<Dims>& indices,
  Dim&&                   dim,
  Loader&&                loader,
  Args&&...               args) noexcept -> void {
  if (indices.is_front(dim)) {
    loader.load_front(
      it.offset(dim, -it.padding()),
      indices.index(dim),
      dim,
      ripple_forward(args)...);
  } else if (indices.is_back(dim)) {
    loader.load_back(
      it.offset(dim, it.padding()),
      indices.index(dim),
      dim,
      ripple_forward(args)...);
  }
}

/**
 * Loads the padding data for the x dimension. This simply forwards to the
 * loading implementation for the x dimension.
 *
 * \param  dim      The overload specifier for a single dimension.
 * \param  it       The iterator to load the padding for.
 * \param  indices  The indices for the ghost cells.
 * \param  loader   The loader to use to load the cells.
 * \param  args     Additional arguments for the loader.
 * \tparam It       The type of the iterator.
 * \tparam Dims     The number of dimensions for the indices.
 * \tparam Loader   The type of the loader for the dims.
 * \tparam Dim      The dimension to load in.
 * \tparam Args     The type of additional arguments.
 */
template <typename It, size_t Dims, typename Loader, typename... Args>
ripple_all auto load_global_padding(
  DimX                    dim,
  It&&                    it,
  const GhostIndex<Dims>& indices,
  Loader&&                loader,
  Args&&... args) noexcept -> void {
  load_global_padding_for_dim(
    it, indices, dim, ripple_forward(loader), ripple_forward(args)...);
}

/**
 * Overload of global loading for a two dimensional case.
 *
 * For two dimensions, corner cells need to load the ghost data in the x,y and
 * xy directions, so this performs the x load, then the y load, and then
 * offsets the it in the y dimension to the y loaded ghost cell, before
 * performing an x load from the ghost cell, to load in the xy element.
 *
 * \param  dim      The overload specifier for two dimensions.
 * \param  it       The iterator to load the padding for.
 * \param  indices  The indices for the ghost cells.
 * \param  loader   The loader to use to load the cells.
 * \param  args     Additional arguments for the loader.
 * \tparam It       The type of the iterator.
 * \tparam Dims     The number of dimensions for the indices.
 * \tparam Loader   The type of the loader for the dims.
 * \tparam Dim      The dimension to load in.
 * \tparam Args     The type of additional arguments.
 */
template <typename It, size_t Dims, typename Loader, typename... Args>
ripple_all auto load_global_padding(
  DimY                    dim,
  It&&                    it,
  const GhostIndex<Dims>& indices,
  Loader&&                loader,
  Args&&... args) noexcept -> void {
  load_global_padding_for_dim(
    it, indices, dimx(), ripple_forward(loader), ripple_forward(args)...);
  load_global_padding_for_dim(
    it, indices, dimy(), ripple_forward(loader), ripple_forward(args)...);

  /* Load corner boundaries: first offset into y padding, then load x
   * boundaries using the padding data for y. The sign of the index is opposite
   * to the direction in which we need to offset because the index normal points
   * into the domain, and here we need to walk out the domaim. */
  const int step = -math::sign(indices.index(dimy())) * it.padding();
  load_global_padding_for_dim(
    it.offset(dimy(), step),
    indices,
    dimx(),
    ripple_forward(loader),
    ripple_forward(args)...);
}

/**
 * Overload of global loading for a three dimensional case. 
 *
 * For three dimensions, corner cells need to load the ghost data in the x,y 
 * and xy directions (i.e a 2D load), as well as the z, zx, zy, zxy directions 
 * (i.e, a z load, then a 2D load from the z loaded cell) so this performs the 
 * 2D load from the \p it cell, then performs the z load, then offsets to the z 
 * loaded cell and performs a 2D load from there.
 *
 * In the worst case (i.e corner cells) there are therefore 7 loads performed
 * by a cell. However, the benefit of this method is that due to the ordering,
 * there is no synchronization required.
 *
 * \param  dim      The overload specifier for three dimensions.
 * \param  it       The iterator to load the padding for.
 * \param  indices  The indices for the ghost cells.
 * \param  loader   The loader to use to load the cells.
 * \param  args     Additional arguments for the loader.
 * \tparam It       The type of the iterator.
 * \tparam Dims     The number of dimensions for the indices.
 * \tparam Loader   The type of the loader for the dims.
 * \tparam Dim      The dimension to load in.
 * \tparam Args     The type of additional arguments.
 */
template <typename It, size_t Dims, typename Loader, typename... Args>
ripple_all auto load_global_padding(
  DimZ                    dim,
  It&&                    it,
  const GhostIndex<Dims>& indices,
  Loader&&                loader,
  Args&&... args) noexcept -> void {
  load_global_padding(
    dimy(), it, indices, ripple_forward(loader), ripple_forward(args)...);

  load_global_padding_for_dim(
    it, indices, dimz(), ripple_forward(loader), ripple_forward(args)...);

  /* Offset in the z dimension and then load the boundaries for the 2D plane
   * from the offset cell. As in the 2D case, the direction to offset in is
   * opposite to the sign of the index since the index normal points into the
   * domain, and here we are moving out of it.
   */
  const int step = -math::sign(indices.index(dimz())) * it.padding();
  load_global_padding(
    dimy(),
    it.offset(dimz(), step),
    indices,
    ripple_forward(loader),
    ripple_forward(args)...);
}

} // namespace ripple::detail

#endif // RIPPLE_PADDING_DETAIL_LOAD_GLOBAL_PADDING_IMPL__HPP

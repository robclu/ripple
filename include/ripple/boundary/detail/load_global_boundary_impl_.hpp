//==--- ../boundary/detail/load_global_boundary_impl_.hpp -- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  load_global_boundary_impl_.hpp
/// \brief This file implements functionality to load global data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_DETAIL_LOAD_GLOBAL_BOUNDARY_IMPL__HPP
#define RIPPLE_BOUNDARY_DETAIL_LOAD_GLOBAL_BOUNDARY_IMPL__HPP

#include "../ghost_index.hpp"

namespace ripple::detail {

/// Loads the boundary data for the \p dim dimension, using the index in the \p
/// dim dimension from the \p indices, and the \p loader to load the data.
/// \param  it       The iterator to load the padding for.
/// \param  indices  The indices for the ghost cells.
/// \param  dim      The dimension to load the cell in.
/// \param  loader   The loader to use to load the cells.
/// \param  args     Additional arguments for the loader.
/// \tparam Iterator The type of the iterator.
/// \tparam Dims     The number of dimensions for the indices.
/// \tparam Loader   The type of the loader for the dims.
/// \tparam Dim      The dimension to load in.
/// \tparam Args     The type of additional arguments.
template <
  typename    Iterator,
  std::size_t Dims    ,
  typename    Loader  ,
  typename    Dim     ,
  typename... Args
>
ripple_host_device auto load_global_boundary_for_dim(
  Iterator&&              it     ,
  const GhostIndex<Dims>& indices,
  Dim&&                   dim    ,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  if (indices.is_front(dim)) {
    loader.load_front(
      it.offset(dim, -it.padding()), indices.index(dim), dim, args...
    );
  } else if (indices.is_back(dim)) {
    loader.load_back(
      it.offset(dim, it.padding()), indices.index(dim), dim, args...
    );
  }
}

/// Loads the boundary data for the x dimension. This simply forwards to the
/// loading implementation for the x dimension.
///
/// \param  dim      The overload specifier for a single dimension.
/// \param  it       The iterator to load the padding for.
/// \param  indices  The indices for the ghost cells.
/// \param  loader   The loader to use to load the cells.
/// \param  args     Additional arguments for the loader.
/// \tparam Iterator The type of the iterator.
/// \tparam Dims     The number of dimensions for the indices.
/// \tparam Loader   The type of the loader for the dims.
/// \tparam Dim      The dimension to load in.
/// \tparam Args     The type of additional arguments.
template <
  typename    Iterator,
  std::size_t Dims    ,
  typename    Loader  ,
  typename... Args
>
ripple_host_device auto load_global_boundary(
  dimx_t                  dimx   ,
  Iterator&&              it     ,
  const GhostIndex<Dims>& indices,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  load_global_boundary_for_dim(
    std::forward<Iterator>(it)  ,
    indices                     ,
    dimx                        ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  );
}

/// Overload of global loading for a two dimensional case. For two dimensions,
/// corner cells need to load the ghost data in the x,y and xy directions, so
/// this performs the x load, then the y load, and then offsets the \p it in the
/// y dimension to the y loaded ghost cell, before performing an x load from the
/// ghost cell, to load in the xy element.
///
/// \param  dim      The overload specifier for two dimensions.
/// \param  it       The iterator to load the padding for.
/// \param  indices  The indices for the ghost cells.
/// \param  loader   The loader to use to load the cells.
/// \param  args     Additional arguments for the loader.
/// \tparam Iterator The type of the iterator.
/// \tparam Dims     The number of dimensions for the indices.
/// \tparam Loader   The type of the loader for the dims.
/// \tparam Dim      The dimension to load in.
/// \tparam Args     The type of additional arguments.
template <
  typename    Iterator,
  std::size_t Dims    ,
  typename    Loader  ,
  typename... Args
>
ripple_host_device auto load_global_boundary(
  dimy_t                  dim    ,
  Iterator&&              it     ,
  const GhostIndex<Dims>& indices,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  // Load x boundary:
  load_global_boundary_for_dim(
    std::forward<Iterator>(it)  ,
    indices                     ,
    dim_x                       ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  );

  // Load y boundary:
  load_global_boundary_for_dim(
    std::forward<Iterator>(it)  ,
    indices                     ,
    dim_y                       ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  );

  // Load corner boundaries, first offset into y padding, then load x
  // boundaries using the padding data for y. The sign of the index is opposite
  // to the direction in which we need to offset because the index normal points
  // into the domain, and here we need to walk out the domaim.
  const auto step = -math::sign(indices.index(dim_y)) * it.padding();
   load_global_boundary_for_dim(
    it.offset(dim_y, step)      ,
    indices                     ,
    dim_x                       ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  ); 
}

/// Overload of global loading for a three dimensional case. For three
/// dimensions, corner cells need to load the ghost data in the x,y and xy
/// directions (i.e a 2D load), as well as the z, zx, zy, zxy directions (i.e, a
/// z load, then a 2D load from the z loaded cell) so this performs the 2D load
/// from the \p it cell, then performs the z load, then offsets to the z loaded
/// cell and performs a 2D load from there.
///
/// In the worst case (i.e corner cells) there are therefore 7 loads performed
/// by a cell. However, the benefit of this method is that due to the ordering,
/// there is no synchronization required.
///
/// \param  dim      The overload specifier for three dimensions.
/// \param  it       The iterator to load the padding for.
/// \param  indices  The indices for the ghost cells.
/// \param  loader   The loader to use to load the cells.
/// \param  args     Additional arguments for the loader.
/// \tparam Iterator The type of the iterator.
/// \tparam Dims     The number of dimensions for the indices.
/// \tparam Loader   The type of the loader for the dims.
/// \tparam Dim      The dimension to load in.
/// \tparam Args     The type of additional arguments.
template <
  typename    Iterator,
  std::size_t Dims    ,
  typename    Loader  ,
  typename... Args>
ripple_host_device auto load_global_boundary(
  dimz_t                  dim    ,
  Iterator&&              it     ,
  const GhostIndex<Dims>& indices,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  // Load boundaries for 2D plane from cell:
  load_global_boundary(
    dim_y                       ,
    std::forward<Iterator>(it)  ,
    indices                     ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  );

  // Set the z boundary for the cell:
  load_global_boundary_for_dim(
    std::forward<Iterator>(it)  ,
    indices                     ,
    dim_z                       ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  );

  // Offset in the z dimension and then load the boundaries for the 2D plane
  // from the offset cell. As in the 2D case, the direction to offset in is
  // opposite to the sign of the index since the index normal points into the
  // domain, and here we are moving out of it.
  const auto step = -math::sign(indices.index(dim_z)) * it.padding();
  load_global_boundary(
    dim_y                       ,
    it.offset(dim_z, step)      ,
    indices                     ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)... 
  );
}

} // namespace ripple::detail


#endif // RIPPLE_BOUNDARY_DETAIL_LOAD_GLOBAL_BOUNDARY_IMPL__HPP
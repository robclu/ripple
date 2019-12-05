//==--- ../boundary/detail/load_internal_boundary_impl_.hpp  -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  load_internal_boundary_impl_.hpp
/// \brief This file implements functionality to load internal boundary data
///        from one iterator to another.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_DETAIL_LOAD_INTERNAL_BOUNDARY_IMPL__HPP
#define RIPPLE_BOUNDARY_DETAIL_LOAD_INTERNAL_BOUNDARY_IMPL__HPP

#include "../ghost_index.hpp"

namespace ripple::detail {

/// Loads the boundary data for the \p dim dimension, using the index in the \p
/// dim dimension from the \p indices, and the \p loader to load the data.
/// \param  it_from      The iterator to load the boundary data from.
/// \param  it_to        The iterator to load the boundary data from.
/// \param  indices      The indices for the ghost cells.
/// \param  dim          The dimension to load the cell in.
/// \param  loader       The loader to use to load the cells.
/// \param  args         Additional arguments for the loader.
/// \tparam IteratorFrom The type of the from iterator.
/// \tparam IteratorTo   The type of the to iterator.
/// \tparam Dims         The number of dimensions for the indices.
/// \tparam Loader       The type of the loader for the dims.
/// \tparam Dim          The dimension to load in.
/// \tparam Args         The type of additional arguments.
template <
  typename    IteratorFrom,
  typename    IteratorTo  ,
  std::size_t Dims        ,
  typename    Loader      ,
  typename    Dim         ,
  typename... Args
>
ripple_host_device auto load_internal_boundary_for_dim(
  IteratorFrom&&          it_from,
  IteratorTo&&            it_to  ,
  const GhostIndex<Dims>& indices,
  Dim&&                   dim    ,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  const auto offset = 
    indices.is_front(dim) ? -it_to.padding() : it_to.padding();
  loader.load(
    it_from.offset(dim, offset),
    it_to.offset(dim, offset)  ,
    std::forward<Args>(args)...
  );
}

/// Loads the boundary data for the x dimension. This simply forwards to the
/// loading implementation for the x dimension after offsetting the iterators.
///
/// \param  dim          The overload specifier for a single dimension.
/// \param  it_from      The iterator to load the boundary data from.
/// \param  it_to        The iterator to load the boundary data to.
/// \param  indices      The indices for the ghost cells.
/// \param  loader       The loader to use to load the cells.
/// \param  args         Additional arguments for the loader.
/// \tparam IteratorFrom The type of the from iterator.
/// \tparam IteratorTo   The type of the to iterator.
/// \tparam Dims         The number of dimensions for the indices.
/// \tparam Loader       The type of the loader for the dims.
/// \tparam Dim          The dimension to load in.
/// \tparam Args         The type of additional arguments.
template <
  typename    IteratorFrom,
  typename    IteratorTo  ,
  std::size_t Dims        ,
  typename    Loader      ,
  typename... Args
>
ripple_host_device auto load_internal_boundary(
  dimx_t                  dimx   ,
  IteratorFrom&&          it_from,
  IteratorTo&&            it_to  ,
  const GhostIndex<Dims>& indices,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  load_internal_boundary_for_dim(
    std::forward<IteratorFrom>(it_from),
    std::forward<IteratorTo>(it_to)    ,
    indices                            ,
    dimx                               ,
    std::forward<Loader>(loader)       ,
    std::forward<Args>(args)...
  );
}

/// Overload of internal loading for a two dimensional case. For two dimensions,
/// corner cells need to load the ghost data in the x,y and xy directions, so
/// this performs the x load, then the y load, and then offsets the iterators in
/// the x and y dimension dimension 
/// the ghost cell, to load in the xy element.
///
/// \param  dim          The overload specifier for two dimensions.
/// \param  it_from      The iterator to load the boundary data from.
/// \param  it_to        The iterator to load the boundary data to.
/// \param  indices      The indices for the ghost cells.
/// \param  loader       The loader to use to load the cells.
/// \param  args         Additional arguments for the loader.
/// \tparam IteratorFrom The type of the from iterator.
/// \tparam IteratorTo   The type of the to iterator.
/// \tparam Dims         The number of dimensions for the indices.
/// \tparam Loader       The type of the loader for the dims.
/// \tparam Dim          The dimension to load in.
/// \tparam Args         The type of additional arguments.
template <
  typename    IteratorFrom,
  typename    IteratorTo  ,
  std::size_t Dims        ,
  typename    Loader      ,
  typename... Args
>
ripple_host_device auto load_internal_boundary(
  dimy_t                  dim    ,
  IteratorFrom&&          it_from,
  IteratorTo&&            it_to  ,
  const GhostIndex<Dims>& indices,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  // Load x boundary:
  load_internal_boundary_for_dim(
    std::forward<IteratorFrom>(it_from),
    std::forward<IteratorTo>(it_to)    ,
    indices                            ,
    dim_x                              ,
    std::forward<Loader>(loader)       ,
    std::forward<Args>(args)...
  );

  // Load y boundary:
  load_internal_boundary_for_dim(
    std::forward<IteratorFrom>(it_from),
    std::forward<IteratorTo>(it_to)    ,
    indices                            ,
    dim_y                              ,
    std::forward<Loader>(loader)       ,
    std::forward<Args>(args)...
  );

  // Load corner boundaries, first offset into y padding, then load x
  // boundaries using the padding data for y. The sign of the index is opposite
  // to the direction in which we need to offset because the index normal points
  // into the domain, and here we need to walk out the domaim.
  const auto step = -math::sign(indices.index(dim_y)) * it_to.padding();
   load_internal_boundary_for_dim(
    it_from.offset(dim_y, step) ,
    it_to.offset(dim_y, step)   ,
    indices                     ,
    dim_x                       ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)...
  ); 
}

/// Overload of internal loading for a three dimensional case. For three
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
/// \param  dim          The overload specifier for three dimensions.
/// \param  it_from      The iterator to load the boundary data from.
/// \param  it_to        The iterator to load the boundary data to.
/// \param  indices      The indices for the ghost cells.
/// \param  loader       The loader to use to load the cells.
/// \param  args         Additional arguments for the loader.
/// \tparam IteratorFrom The type of the from iterator.
/// \tparam IteratorTo   The type of the to iterator.
/// \tparam Dims         The number of dimensions for the indices.
/// \tparam Loader       The type of the loader for the dims.
/// \tparam Dim          The dimension to load in.
/// \tparam Args         The type of additional arguments.
template <
  typename    IteratorFrom,
  typename    IteratorTo  ,
  std::size_t Dims        ,
  typename    Loader      ,
  typename... Args
>
ripple_host_device auto load_internal_boundary(
  dimz_t                  dim    ,
  IteratorFrom&&          it_from,
  IteratorTo&&            it_to  ,
  const GhostIndex<Dims>& indices,
  Loader&&                loader ,
  Args&&...               args
) -> void {
  // Load boundaries for 2D plane from cell:
  load_internal_boundary(
    dim_y                              ,
    std::forward<IteratorFrom>(it_from),
    std::forward<IteratorTo>(it_to)    ,
    indices                            ,
    std::forward<Loader>(loader)       ,
    std::forward<Args>(args)...
  );

  // Set the z boundary for the cell:
  load_internal_boundary_for_dim(
    std::forward<IteratorFrom>(it_from),
    std::forward<IteratorTo>(it_to)    ,
    indices                            ,
    dim_z                              ,
    std::forward<Loader>(loader)       ,
    std::forward<Args>(args)...
  );

  // Offset in the z dimension and then load the boundaries for the 2D plane
  // from the offset cell. As in the 2D case, the direction to offset in is
  // opposite to the sign of the index since the index normal points into the
  // domain, and here we are moving out of it.
  const auto step = -math::sign(indices.index(dim_z)) * it_to.padding();
  load_internal_boundary(
    dim_y                       ,
    it_from.offset(dim_z, step) ,
    it_to.offset(dim_z, step)   ,
    indices                     ,
    std::forward<Loader>(loader),
    std::forward<Args>(args)... 
  );
}

} // namespace ripple::detail


#endif // RIPPLE_BOUNDARY_DETAIL_LOAD_GLOBAL_BOU

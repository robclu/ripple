//==--- ripple/core/container/memcpy_padding.hpp ----------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  memcpy_padding.hpp
/// \brief This file defines a utilities for copying padding between blocks.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_MEMCOPY_PADDING_HPP
#define RIPPLE_CONTAINER_MEMCOPY_PADDING_HPP

#include "block_traits.hpp"

namespace ripple {

//==--- [faces between blocks] ---------------------------------------------==//

/// Defines the location of a face.
enum class Face : uint8_t {
  start = 0, //< Face at the start of the domain.
  end   = 1  //!< Face at the end of the domain.
};

/// Defines the type of mapping for a copy.
enum class Mapping : int {
  domain  = 0, //!< Mapping to copy to/from inside the domain.
  padding = 1  //!< Mapping to copy to/from inside the padding
};

/// Specifies a face for a dimension.
/// \param Dim      The dimension of the face.
/// \param Location The location of the face.
template <size_t Dim, Face Location, Mapping Map = Mapping::domain>
struct FaceSpecifier {};

//==--- [face specifier aliases] -------------------------------------------==//

/// Alias for a face in the x dimension with a given location.
/// \tparam Location The location of the face in the dimension.
template <Face Location>
using x_face_t = FaceSpecifier<dimx_t::value, Location>;

/// Alias for a face in the y dimension with a given location.
/// \tparam Location The location of the face in the dimension.
template <Face Location>
using y_face_t = FaceSpecifier<dimy_t::value, Location>;

//==--- [padding utilites] -------------------------------------------------==//

/// Returns a pointer to the start of the padding for the block when the
/// face padding is the face given by Location in the Dim dimension.
/// \param  block    The block to get the padding pointer for.
/// \tparam Block    The type of the block.
/// \tparam Dim      The dimension in which to get the padding pointer.
/// \tparam Location The destination face for the padding.
template <typename Block, size_t Dim, Face Location, Mapping Map>
auto padding_ptr(Block& block, FaceSpecifier<Dim, Location, Map>) noexcept
  -> decltype(block.begin().data()) {
  constexpr auto dim = Dimension<Dim>();
  if constexpr (Location == Face::start) {
    // If we want a pointer into the padding then we need to offset by the
    // amount of the padding:
    const int shift = (Map == Mapping::padding ? -1 : 0) * block.padding();
    return block.begin().offset(dim, shift).data();
  } else {
    // If we want a pointer into the domain, then we need to subtract the
    // padding amount from the size of the domain, otherwise we subtract 0:
    const int shift =
      block.size(dim) - block.padding() * (Map == Mapping::domain ? 1 : 0);
    return block.begin().offset(dim, shift).data();
  }
}

//==--- [1D face padding copy] ---------------------------------------------==//

/// Copies the relevant padding data from \p src_block into the destination
/// block \p dst_block.
///
/// Src is the location of the face from \p src_block to copy __from__
/// while Dst is the location of the face in \p dst_block to copy __into__.
///
/// If \p dst_block doesn't have any padding then this simply returns.
///
/// This overload is only enabled when the block is 1 dimensional.
///
/// \param  src_block The source block to copy padding from.
/// \param  dst_block The destination block to copy padding into.
/// \param  src_face  The source face for the padding.
/// \param  dst_face  The destination face for the padding.
/// \tparam SrcBlock  The type of the source block.
/// \tparam DstBlock  The type of the destination block.
/// \tparam Src       The location of the source face.
/// \tparam Dst       The location of the destination face.
template <
  typename SrcBlock,
  typename DstBlock,
  Face Src,
  Face Dst,
  block_1d_enable_t<SrcBlock> = 0>
auto memcopy_padding(
  const SrcBlock& src_block,
  DstBlock&       dst_block,
  x_face_t<Src>   src_face,
  x_face_t<Dst>   dst_face) -> void {
  if (dst_block.padding() == 0) {
    return;
  }
  using allocator_t = typename block_traits_t<SrcBlock>::Allocator;

  const auto  copy_type = src_block.template get_copy_type<DstBlock>();
  const auto  copy_size = allocator_t::allocation_size(dst_block.padding());
  const void* src_ptr   = padding_ptr(src_block, src_face);
  void*       dst_ptr   = padding_ptr(dst_block, dst_face);

  cudaMemcpyAsync(
    dst_ptr,
    src_ptr,
    copy_size,
    copy_type,
    is_device_block_v<SrcBlock>
      ? src_block.stream()
      : is_device_block_v<DstBlock> ? dst_block.stream() : 0);
}

//==--- [2D face padding copy] ---------------------------------------------==//

/// Copies the relevant padding data from \p src_block into the destination
/// block \p dst_block.
///
/// The Src is the location of the face from \p src_block to copy from
/// while Dst is the location of the face in \p dst_block to copy __into__.
///
/// If destination block doesn't have padding, this simply returns.
///
/// This overload is only enabled when the block is 2 dimensional.
///
/// \param  src_block The block to copy padding data from.
/// \param  dst_block The block to copy padding into.
/// \param  src_face  The source face for the padding.
/// \param  dst_face  The destination face for the padding.
/// \param  stream    The stream to perform the copy on.
/// \tparam SrcBlock  The type of the source block.
/// \tparam DstBlock  The type of the destination block.
/// \tparam Dim       The dimension in which to copy padding data.
/// \tparam Src       The location of the source face.
/// \tparam Dst       The location of the destination face.
template <
  typename SrcBlock,
  typename DstBlock,
  size_t  Dim,
  Face    Src,
  Face    Dst,
  Mapping SrcMap,
  Mapping DstMap,
  block_2d_enable_t<SrcBlock> = 0>
auto memcopy_padding(
  const SrcBlock&                 src_block,
  DstBlock&                       dst_block,
  FaceSpecifier<Dim, Src, SrcMap> src_face,
  FaceSpecifier<Dim, Dst, DstMap> dst_face,
  cudaStream_t                    stream = 0) -> void {
  using Allocator = typename block_traits_t<SrcBlock>::Allocator;
  static_assert(Dim <= dimy_t::value, "Invalid dimension!");
  if (dst_block.padding() == 0) {
    return;
  }

  const auto   type      = src_block.template get_copy_type<DstBlock>();
  const size_t byte_size = Allocator::allocation_size(1);
  const size_t pitch     = byte_size * src_block.pitch(dim_x);
  const void*  src_ptr   = padding_ptr(src_block, src_face);
  void*        dst_ptr   = padding_ptr(dst_block, dst_face);

  // If copying in x dimension, we just need to copy the padding width,
  // otherwise if in the y dimension, we need to copy the whole width of
  // the x dimension:
  const size_t width =
    byte_size * (Dim == dim_x ? src_block.padding() : src_block.size(dim_x));

  // If copying in the y dimension it is the opposite as above. We need to
  // copy the whole size of the y dimension for the height if the face is
  // in the x dimension, otherwise the height is the amount of padding.
  const size_t height = Dim == dim_x ? src_block.size(dim_y)
                                     : src_block.padding();

  cudaMemcpy2DAsync(
    dst_ptr, pitch, src_ptr, pitch, width, height, type, stream);
}

} // namespace ripple

#endif // RIPPLE_CONTAINER_MEMCOPY_PADDING_HPP
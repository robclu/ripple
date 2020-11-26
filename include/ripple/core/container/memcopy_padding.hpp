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

/*==--- [faces between blocks] ---------------------------------------------==*/

/**
 * Defines the location of a face.
 */
enum class FaceLocation : uint8_t {
  start = 0, //< Face at the start of the domain.
  end   = 1  //!< Face at the end of the domain.
};

/**
 * Defines the type of mapping for a copy -- whether the copy uses data from
 * a padding region or a region inside the domain.
 */
enum class Mapping : int {
  domain  = 0, //!< Mapping to copy to/from inside the domain.
  padding = 1  //!< Mapping to copy to/from inside the padding
};

/**
 * Specifies the type of a copy to be performed.
 * \tparam Dim      The dimension of the face.
 * \tparam Location The location of the face.
 * \tparam Mapping  The mapping for the copy.
 */
template <size_t Dim, FaceLocation Location, Mapping Map = Mapping::domain>
struct CopySpecifier {};

/*==--- [copy specifier aliases] -------------------------------------------==*/

/**
 * Alias for a copy from a face in x dimension with a given location.
 * \tparam Location The location of the face in the dimension.
 */
template <FaceLocation Location>
using CopySpecifierX = CopySpecifier<dimx_t::value, Location>;

/**
 * Alias for a copy from a face in the y dimension with a given location.
 * \tparam Location The location of the face in the dimension.
 */
template <FaceLocation Location>
using CopySpecifierY = CopySpecifier<dimy_t::value, Location>;

/*==--- [padding utilites] -------------------------------------------------==*/

/**
 * Parameters for offseting the iterator to the storage.
 */
struct OffsetParam {
  int dimension = -1; //!< The dimension to offset in.
  int amount    = 0;  //!< The amount to offset.
};

/**
 * Gets a pointer to the start of the padding for the block for the given copy
 * spcification.
 *
 * \note If the iterated type is a storage accessor and therefore could be
 *       strided or contiguous, then the underlying storage type is returned,
 *       otherwise a raw pointer to the data is returned.
 *
 * \param  block    The block to get the padding pointer for.
 * \param  offset   An additional offset for the pointer.
 * \tparam Block    The type of the block.
 * \tparam Dim      The dimension in which to get the padding pointer.
 * \tparam Location The destination face for the padding.
 * \tparam Map      The mapping for the copy.
 */
template <typename Block, size_t Dim, FaceLocation Location, Mapping Map>
auto storage_ptr(
  Block& block,
  CopySpecifier<Dim, Location, Map>,
  OffsetParam offset = OffsetParam()) noexcept {
  constexpr auto dim   = Dimension<Dim>();
  constexpr bool pad   = Map == Mapping::padding;
  int            shift = 0;
  if constexpr (Location == FaceLocation::start) {
    /* If a pointer into the padding is required then we need to offset by the
     * amount of the padding, since the iterator points to the first valid cell.
     */
    shift = (pad ? -1 : 0) * block.padding();
  } else {
    /* If a pointer into the domain is required then we need to subtract the
     * padding amount from the size of the domain, since the size of the block
     * given the end of the domain (or start of the padding):
     */
    shift = block.size(dim) - block.padding() * (pad ? 0 : 1);
  }
  auto it = block.begin().offset(dim, shift);
  if (offset.dimension != -1) {
    it.shift(offset.dimension == 0 ? dimx() : dimy(), offset.amount);
  }

  if constexpr (is_storage_accessor_v<decltype(it.storage())>) {
    return it.storage();
  } else {
    return it.data();
  }
}

/*==--- [1D face padding copy] ---------------------------------------------==*/

/**
 * Copies the relevant padding data from the source block into the destination
 * block.
 *
 * Src is the location of the face from \p src_block to copy __from__
 * while Dst is the location of the face in \p dst_block to copy __into__.
 *
 * If \p dst_block doesn't have any padding then this simply returns.
 *
 * This overload is only enabled when the block is 1 dimensional.
 *
 * \param  src_block The source block to copy padding from.
 * \param  dst_block The destination block to copy padding into.
 * \param  src_face  The source face for the padding.
 * \param  dst_face  The destination face for the padding.
 * \tparam SrcBlock  The type of the source block.
 * \tparam DstBlock  The type of the destination block.
 * \tparam Src       The location of the source face.
 * \tparam Dst       The location of the destination face.
 */
template <
  typename SrcBlock,
  typename DstBlock,
  FaceLocation SrcLocation,
  FaceLocation DstLocation,
  block_1d_enable_t<SrcBlock> = 0>
auto memcopy_padding(
  const SrcBlock&             src_block,
  DstBlock&                   dst_block,
  CopySpecifierX<SrcLocation> src_specifier,
  CopySpecifierX<DstLocation> dst_specifier) -> void {
  if (dst_block.padding() == 0) { return; }
  using Allocator = typename block_traits_t<SrcBlock>::Allocator;

  const auto  copy_type = src_block.template get_copy_type<DstBlock>();
  const auto  copy_size = Allocator::allocation_size(dst_block.padding());
  const void* src_ptr   = padding_ptr(src_block, src_specifier);
  void*       dst_ptr   = padding_ptr(dst_block, dst_specifier);

  ripple_if_cuda(cudaMemcpyAsync(
    dst_ptr,
    src_ptr,
    copy_size,
    copy_type,
    is_device_block_v<SrcBlock>   ? src_block.stream()
    : is_device_block_v<DstBlock> ? dst_block.stream()
                                  : 0));
}

/*==--- [2D face padding copy] ---------------------------------------------==*/

/**
 * Copies the relevant padding data from the source block into the destination
 * block.
 *
 * The source location is the location of the face to copy from while the
 * des location is the location of the face to copy __into__.
 *
 * \note If destination block doesn't have padding, this simply returns.
 *
 * \note This overload is only enabled when the block is 2 dimensional.
 *
 * \param  src_block     The block to copy padding data from.
 * \param  dst_block     The block to copy padding into.
 * \param  src_specifier The source face for the padding.
 * \param  dst_specifier The destination face for the padding.
 * \param  stream        The stream to perform the copy on.
 * \tparam SrcBlock      The type of the source block.
 * \tparam DstBlock      The type of the destination block.
 * \tparam Dim           The dimension in which to copy padding data.
 * \tparam SrcLocation   The location of the source face.
 * \tparam DstLocation   The location of the destination face.
 * \tparam SrcMapping    The mapping for the source.
 * \taram  DstMappding   The mapping for the destination.
 */
template <
  typename SrcBlock,
  typename DstBlock,
  size_t       Dim,
  FaceLocation SrcLocation,
  FaceLocation DstLocation,
  Mapping      SrcMapping,
  Mapping      DstMapping,
  block_2d_enable_t<SrcBlock> = 0>
auto memcopy_padding(
  const SrcBlock&                             src_block,
  DstBlock&                                   dst_block,
  CopySpecifier<Dim, SrcLocation, SrcMapping> src_specifier,
  CopySpecifier<Dim, DstLocation, DstMapping> dst_specifier,
  GpuStream                                   stream = 0) -> void {
  using Allocator = typename block_traits_t<SrcBlock>::Allocator;
  static_assert(Dim <= dimy_t::value, "Invalid dimension!");
  if (dst_block.padding() == 0) { return; }

  OffsetParam p{
    Dim == dimx() ? dimy_t::value : dimx_t::value,
    -1 * static_cast<int>(src_block.padding())};

  constexpr size_t num_types = Allocator::num_different_types();
  constexpr bool   is_accessor =
    is_storage_accessor_v<decltype(storage_ptr(src_block, src_specifier, p))>;

  std::vector<const void*> src_ptrs;
  std::vector<void*>       dst_ptrs;
  /* If the storage is a storage accessor then it may be strided and have
   * multiple types for which the memory needs to be copied, so here we create
   * vector of all the pointers to copy, from which we can then do the generic
   * implementation. */
  if constexpr (is_accessor) {
    for (auto* p : storage_ptr(src_block, src_specifier, p).data_ptrs()) {
      src_ptrs.push_back(p);
    }
    for (auto* p : storage_ptr(dst_block, dst_specifier, p).data_ptrs()) {
      dst_ptrs.push_back(p);
    }
  } else {
    src_ptrs.push_back(storage_ptr(src_block, src_specifier, p));
    dst_ptrs.push_back(storage_ptr(dst_block, dst_specifier, p));
  }

  const auto type = src_block.template get_copy_type<DstBlock>();
  unrolled_for<num_types>([&](auto i) {
    constexpr size_t bytes = Allocator::template byte_size<i>();
    constexpr size_t elems = Allocator::template num_elements<i>();

    /* Pitch is always the number of elements (including padding elements)
     * multiplied by the number of bytes. */
    const size_t pitch = bytes * src_block.pitch(dimx());

    /* If copying in x dimension, we just need to copy the padding width,
     * otherwise if in the y dimension, we need to copy the whole width of
     * the x dimension: */
    const size_t width =
      bytes * (Dim == dimx() ? src_block.padding() : src_block.pitch(dimx()));

    /* If copying in the y dimension it is the opposite as above. We need to
     * copy the whole size of the y dimension for the height if the face is
     * in the x dimension, otherwise the height is the amount of padding.
     *
     * For the case that the data is strided, then we are using the pitch for
     * a single element, so we need to multiply by the number of elements to
     * copy all the data. */
    const size_t height =
      (Dim == dimx() ? src_block.pitch(dimy()) : src_block.padding()) * elems;

    ripple_check_cuda_result(ripple_if_cuda(cudaMemcpy2DAsync(
      dst_ptrs[i], pitch, src_ptrs[i], pitch, width, height, type, stream)));
  });
}

/**
 * Copies the relevant padding data from the source block into the destination
 * block.
 *
 * The source location is the location of the face to copy from while the
 * des location is the location of the face to copy __into__.
 *
 * \note If destination block doesn't have padding, this simply returns.
 *
 * \note This overload is only enabled when the block is 3 dimensional.
 *
 * \param  src_block     The block to copy padding data from.
 * \param  dst_block     The block to copy padding into.
 * \param  src_specifier The source face for the padding.
 * \param  dst_specifier The destination face for the padding.
 * \param  stream        The stream to perform the copy on.
 * \tparam SrcBlock      The type of the source block.
 * \tparam DstBlock      The type of the destination block.
 * \tparam Dim           The dimension in which to copy padding data.
 * \tparam SrcLocation   The location of the source face.
 * \tparam DstLocation   The location of the destination face.
 * \tparam SrcMapping    The mapping for the source.
 * \taram  DstMappding   The mapping for the destination.
 */
template <
  typename SrcBlock,
  typename DstBlock,
  size_t       Dim,
  FaceLocation SrcLocation,
  FaceLocation DstLocation,
  Mapping      SrcMapping,
  Mapping      DstMapping,
  block_3d_enable_t<SrcBlock> = 0>
auto memcopy_padding(
  const SrcBlock&                             src_block,
  DstBlock&                                   dst_block,
  CopySpecifier<Dim, SrcLocation, SrcMapping> src_specifier,
  CopySpecifier<Dim, DstLocation, DstMapping> dst_specifier,
  GpuStream                                   stream = 0) -> void {
  using Allocator = typename block_traits_t<SrcBlock>::Allocator;
  static_assert(Dim <= dimz_t::value, "Invalid dimension!");
  if (dst_block.padding() == 0) { return; }

  OffsetParam p{
    Dim == dimx() ? dimy_t::value : dimx_t::value,
    -1 * static_cast<int>(src_block.padding())};

  constexpr size_t num_types = Allocator::num_different_types();
  constexpr bool   is_accessor =
    is_storage_accessor_v<decltype(storage_ptr(src_block, src_specifier, p))>;

  std::vector<const void*> src_ptrs;
  std::vector<void*>       dst_ptrs;
  /* If the storage is a storage accessor then it may be strided and have
   * multiple types for which the memory needs to be copied, so here we create
   * vector of all the pointers to copy, from which we can then do the generic
   * implementation. */
  if constexpr (is_accessor) {
    for (auto* p : storage_ptr(src_block, src_specifier, p).data_ptrs()) {
      src_ptrs.push_back(p);
    }
    for (auto* p : storage_ptr(dst_block, dst_specifier, p).data_ptrs()) {
      dst_ptrs.push_back(p);
    }
  } else {
    src_ptrs.push_back(storage_ptr(src_block, src_specifier, p));
    dst_ptrs.push_back(storage_ptr(dst_block, dst_specifier, p));
  }

  const auto type = src_block.template get_copy_type<DstBlock>();
  unrolled_for<num_types>([&](auto i) {
    constexpr size_t bytes = Allocator::template byte_size<i>();
    constexpr size_t elems = Allocator::template num_elements<i>();

    /* Pitch is always the number of elements (including padding elements)
     * multiplied by the number of bytes. */

    /* This pitch here is tricky. At the very least, the pich is:
     *
     *  - bytes per element * elements in dim x
     *
     * then, if we are copying in y or z, we multiply by the number of elements
     * for the strided case.
     *
     * Lastly, for the z-case, we want to copy padding number of x-y planes,
     * for which the data in the plane is contiguous, and each plane for the
     * padding is also contiguous, so we can just set the pitch to be the
     * entire region to copy, which is better for performance.
     */
    const size_t pitch = bytes * src_block.pitch(dimx()) *
                         (Dim == dimx() ? 1 : elems) *
                         (Dim == dimz() ? src_block.padding() : 1);

    /* If copying in x dimension, we just need to copy the padding width,
     * otherwise if in the y dimension, we need to copy the whole width of
     * the x dimension: */
    const size_t width =
      Dim == dimz() ? pitch
                    : bytes * (Dim == dimx() ? src_block.padding()
                                             : src_block.pitch(dimx()) * elems);

    /* If copying in the y dimension it is the opposite as above. We need to
     * copy the whole size of the y dimension for the height if the face is
     * in the x dimension, otherwise the height is the amount of padding.
     *
     * For the case that the data is strided, then we are using the pitch for
     * a single element, so we need to multiply by the number of elements to
     * copy all the data. */
    const size_t height =
      Dim == dimz() ? 1
                    : src_block.pitch(dimz()) *
                        (Dim != dimx() ? 1 : src_block.pitch(dimy()) * elems);

    ripple_check_cuda_result(ripple_if_cuda(cudaMemcpy2DAsync(
      dst_ptrs[i], pitch, src_ptrs[i], pitch, width, height, type, stream)));
  });
}

} // namespace ripple

#endif // RIPPLE_CONTAINER_MEMCOPY_PADDING_HPP
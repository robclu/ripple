//==--- ripple/multidim/offset_to.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  offset_to.hpp
/// \brief This file imlements functionality for computing the offset to an
///        index in a multidimensional space based on the layout of the data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MULTIDIM_OFFSET_TO_HPP
#define RIPPLE_MULTIDIM_OFFSET_TO_HPP

#include "multidim_space.hpp"
#include <ripple/container/vec.hpp>

namespace ripple {

/// Returns the offset to the element at the index defined by the \p indices for
/// each of the dimension in a space defined by the \p space. The \p
/// element_size is the number of elements in the array type when the data is
/// stored in SoA format. If the data is stored in SoA format, but has a size of
/// 1, this will compute the offset as if the data is stored contiguously.
///
/// For example, for a 2D velocity array with u and v, the layout would be
///
/// ~~~
/// index:    0   1   2   3   ...   n
/// ---     ===========================
///         | u | u | u | u | ... | u |
///  0      ---------------------------
///         | v | v | v | v | ... | v |
/// ---     ===========================
///         | u | u | u | u | ... | u |
///  1      ---------------------------
///         | v | v | v | v | ... | v |
/// ---     ===========================
/// ~~~
///
/// Here \p element_size would be 2, since to compute the offset it's required
/// that the vertical size be know. In the above example, with  `n=10`,
/// `element_size=2`, `indices=(0, 1)` would return an offset of 20, while 
/// `indices=(4, 1)` would return an offset of 24.
///
/// \param  space        The multidimensional space to get the offset in.
/// \param  element_size The size of the elements stored in the space.
/// \param  indices      The indicies for the element in the space.
/// \tparam SpatialImpl  The implementation of the spatial interface.
/// \tparam Indices      The types of the indices.
template <typename SpatialImpl, typename... Indices>
ripple_host_device auto offset_to_soa(
  const MultidimSpace<SpatialImpl>& space       ,
  std::size_t                       element_size,
  Indices&&...                      indices
) -> std::size_t {
  constexpr auto num_indices = sizeof...(Indices);
  const auto     ids         = Vec<std::size_t, num_indices>{indices...};
  std::size_t    offset      = ids[0];
  unrolled_for<num_indices - 1>([&] (auto i) {
    constexpr auto dim = static_cast<std::size_t>(i) + 1;
    offset += ids[dim] * element_size * space.step(dim); 
  });
  return offset;
}

/// Returns the offset to the element at the index defined by the \p indices for
/// each of the dimension in a space defined by the \p space. The \p
/// element_size is the number of elements in the type. For all types, this
/// should be 1, since the types are next to each other, however, when using
/// this to compute the offset when the data is stored as SoA or AoS, \p
/// element_size is the number of components in the type being stored. For
///

/// Returns the offset to the element at the index defined by the \p indices for
/// each of the dimension in a space defined by the \p space. The \p
/// element_size is the number of elements in the array type, if the data stored
/// the underlying type in the array.
///
/// For example, for a 2D velocity array for types T and the underlying storage
/// for the space allocates T, the element_size would be 2. For complex types
/// which store the whole type, for example if Vec<T, 2> is stored rather than
/// raw T, then the \p element_size would be 1. This allows for the caller to
/// compute the offset based on the way the data has been stored.
/// 
/// For the case that the u, v data is allocated as type T, the storage would
/// look as follows:
///
/// ~~~
/// index:  |   0   |   1   | ... |   n   |
///
/// ---     ==============================|
///  0      | u | v | u | v | ... | u | v |
/// ---     ===============================
///  1      | u | v | u | v | ... | u | v |
/// ---     ===============================
///  2      | u | v | u | v | ... | u | v |
/// ---     ===============================
/// ~~~
///
/// Here \p element_size would be 2, since to compute the offset it's required
/// that the horizontal size be know. In the above example, with  `n=10`,
/// `element_size=2`, `indices=(0, 1)` would return an offset of 20, while 
/// `indices=(3, 0)` would return an offset of 6.
///
/// For the case that the u, v data is allocated as type Vec<T, 2>, the storage
/// would look as follows:
///
/// ~~~
/// index:  |     0     |     1     | ... |     n      |
///
/// ---     ============================================
///  0      | Vec<T, 2> | Vec<T, 2> | ... | Vec<>T, 2> |
/// ---     ============================================
///  1      | Vec<T, 2> | Vec<T, 2> | ... | Vec<>T, 2> |
/// ---     ============================================
///  2      | Vec<T, 2> | Vec<T, 2> | ... | Vec<>T, 2> |
/// ---     ============================================
/// ~~~
///
/// Here \p element_size would be 1, since the underlying storage allocated the
/// complex type contiguously. . In the above example, with  `n=10`,
/// `element_size=1`, `indices=(0, 1)` would return an offset of 10, while 
/// `indices=(3, 0)` would return an offset of 3.
///
/// \param  space        The multidimensional space to get the offset in.
/// \param  element_size The size of the elements stored in the space.
/// \param  indices      The indicies for the element in the space.
/// \tparam SpatialImpl  The implementation of the spatial interface.
/// \tparam Indices      The types of the indices.
template <typename SpatialImpl, typename... Indices>
ripple_host_device auto offset_to_aos(
  const MultidimSpace<SpatialImpl>& space       ,
  std::size_t                       element_size,
  Indices&&...                      indices
) -> std::size_t {
  constexpr auto num_indices = sizeof...(Indices);
  const auto     ids         = Vec<std::size_t, num_indices>{indices...};
  std::size_t    offset      = ids[0] * element_size;
  unrolled_for<num_indices - 1>([&] (auto i) {
    constexpr auto dim = static_cast<std::size_t>(i) + 1;
    offset += ids[dim] * element_size * space.step(dim);
  });
  return offset;
}

} // namespace ripple

#endif // RIPPLE_MULTIDIM_OFFSET_TO_HPP

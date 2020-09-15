//==--- ../boundary/detail/load_internal_boundary_impl_.hpp  -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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

#include <ripple/core/utility/type_traits.hpp>

namespace ripple::detail {

// clang-format off
/**
 * Loads the boundary data for Dims dimensions, using \p it_from iterator to
 * copy the data into the \p it_to iterator.
 *
 * This overload is only enabled when Dims = 1.
 *
 * \param  it_from  The iterator to load the boundary data from.
 * \param  it_to    The iterator to load the boundary data to.
 * \tparam Dims     The number of dimensions to load for.
 * \tparam ItFrom   The type of the from iterator.
 * \tparam ItTo     The type of the to iterator.
 */
template <
  size_t Dims, typename IteFrom, typename ItTo, dim_1d_enable_t<Dims> = 0>
ripple_host_device auto
  load_internal(ItFrom&& it_from, ItTo&& it_to) noexcept -> void {
  // clang-format on
  const auto pad      = static_cast<int>(it_to.padding());
  const auto dim_size = static_cast<int>(std::min(
    it_to.size(dim_x),
    it_from.size(dim_x) - block_idx(dim_x) * block_size(dim_x)));

  // Number of iterations for the dimension. Here, this is essentially the ceil
  // of (size + pad) / size:
  const auto iters        = (2 * (dim_size + pad) - 1) / dim_size;
  const auto shift_amount = std::min(dim_size, 2 * pad);
  for (auto i : range(iters)) {
    *it_to.offset(dim_x, i * shift_amount) =
      *it_from.offset(dim_x, i * shift_amount);
  }
}

// clang-format off

/** 
 * Loads the boundary data for Dims dimensions, using \p it_from iterator to
 * copy the data into the \p it_to iterator.
 *
 * This overload is only enabled when Dims > 1, and it recursively calls the
 * implementation from the largest dimension to the smallest dimension.
 *
 * \param  it_from The iterator to load the boundary data from.
 * \param  it_to   The iterator to load the boundary data to.
 * \tparam Dims    The number of dimensions to load for.
 * \tparam ItFrom  The type of the from iterator.
 * \tparam ItTo    The type of the to iterator.
 */
template <
  size_t Dims, typename ItFrom, typename ItTo, not_dim_1d_enable_t<Dims> = 0>
ripple_host_device auto
load_internal(ItFrom&& it_from, ItTo&& it_to) noexcept -> void {
  // Has to be 2d or 3d:
  constexpr auto dim      = Dims == 3 ? dim_z : dim_y;
  const auto     pad      = static_cast<int>(it_to.padding());
  const auto     dim_size = static_cast<int>(std::min(
    it_to.size(dim), it_from.size(dim) - block_idx(dim) * block_size(dim)));

  // Number of iterations for the dimension. Here, this is essentially the ceil
  // of (size + pad) / size:
  const auto iters        = (2 * (dim_size + pad) - 1) / dim_size;
  const auto shift_amount = std::min(dim_size, 2 * pad);
  for (auto i : range(iters)) {
    load_internal<Dims - 1>(
      it_from.offset(dim, i * shift_amount),
      it_to.offset(dim, i * shift_amount));
  }
}

} // namespace ripple::detail

#endif // RIPPLE_BOUNDARY_DETAIL_LOAD_GLOBAL_BOU

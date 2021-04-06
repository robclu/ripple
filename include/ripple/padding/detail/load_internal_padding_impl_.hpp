/**=--- ../padding/detail/load_internal_padding_impl_.hpp -- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  load_internal_padding_impl_.hpp
 * \brief This file implements functionality to load internal padding data.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_DETAIL_LOAD_INTERNAL_PADDING_IMPL__HPP
#define RIPPLE_PADDING_DETAIL_LOAD_INTERNAL_PADDING_IMPL__HPP

#include <ripple/utility/range.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple::detail {

// clang-format off
/**
 * Loads the padding data for Dims dimensions, using the from iterator to
 * copy the data into the to iterator.
 *
 * This overload is only enabled when Dims = 1.
 *
 * \param  from   The iterator to load the boundary data from.
 * \param  to     The iterator to load the boundary data to.
 * \tparam Dims   The number of dimensions to load for.
 * \tparam ItFrom The type of the from iterator.
 * \tparam ItTo   The type of the to iterator.
 */
template <
  size_t Dims, typename ItFrom, typename ItTo, dim_1d_enable_t<Dims> = 0>
ripple_host_device auto
  load_internal(ItFrom&& from, ItTo&& to) noexcept -> void {
  // clang-format on
  const int pad      = static_cast<int>(to.padding());
  const int dim_size = static_cast<int>(std::min(
    to.size(dimx()),
    from.size(dimx()) - block_idx(dimx()) * block_size(dimx())));

  // Number of iterations for the dimension.
  // Here, this is essentially the ceil of (size + pad) / size:
  const int iters        = (2 * (dim_size + pad) - 1) / dim_size;
  const int shift_amount = std::min(dim_size, 2 * pad);
  for (int i : range(iters)) {
    *to.offset(dimx(), i * shift_amount) =
      *from.offset(dimx(), i * shift_amount);
  }
}

// clang-format off

/** 
 * Loads the padding data for Dims dimensions, using the from iterator to
 * copy the data into the to iterator.
 *
 * This overload is only enabled when Dims > 1, and it recursively calls the
 * implementation from the largest dimension to the smallest dimension.
 *
 * \param  from   The iterator to load the boundary data from.
 * \param  to     The iterator to load the boundary data to.
 * \tparam Dims   The number of dimensions to load for.
 * \tparam ItFrom The type of the from iterator.
 * \tparam ItTo   The type of the to iterator.
 */
template <
  size_t Dims, typename ItFrom, typename ItTo, not_dim_1d_enable_t<Dims> = 0>
ripple_host_device auto
load_internal(ItFrom&& from, ItTo&& to) noexcept -> void {
  // clang-format on
  // Has to be 2d or 3d:
  constexpr auto dim      = Dims == 3 ? dimz() : dimy();
  const int      pad      = static_cast<int>(to.padding());
  const int      dim_size = static_cast<int>(
    std::min(to.size(dim), from.size(dim) - block_idx(dim) * block_size(dim)));

  // Number of iterations for the dimension. Here, this is essentially the ceil
  // of (size + pad) / size:
  const int iters        = (2 * (dim_size + pad) - 1) / dim_size;
  const int shift_amount = std::min(dim_size, 2 * pad);
  for (int i : range(iters)) {
    load_internal<Dims - 1>(
      from.offset(dim, i * shift_amount), to.offset(dim, i * shift_amount));
  }
}

} // namespace ripple::detail

#endif // RIPPLE_PADDING_DETAIL_LOAD_INTERNAL_PADDING_IMPL__HPP

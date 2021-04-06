/**=--- ripple/padding/fo_extrap_loader_.hpp --------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  fo_extrap_loader.hpp
 * \brief This file defines an implementation of a padding loader which
 *        performs a first order (constant) extrapolation from the last cell
 *        inside the domain to all padding cells.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_FO_EXTRAP_LOADER_HPP
#define RIPPLE_PADDING_FO_EXTRAP_LOADER_HPP

#include "padding_loader.hpp"

namespace ripple {

/**
 * The FOExtrapLoader is an implementation of an BoundaryLoader which copies
 * the data from the closest valid cell inside the domain to all padding
 * cells.
 *
 * It performs a first order extrapolation of the data in an iterator into the
 * padding data for the iterator.
 */
struct FOExtrapLoader : public PaddingLoader<FOExtrapLoader> {
  /**
   * Loads the front padding in the dim dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the padding cell to load.
   * \param  index    The index of the padding cell in the dimension.
   * \param  dim      The dimension to load the padding in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto
  load_front(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    static_assert_iterator(it);
    *it = *it.offset(dim, index);
  }

  /**
   * Loads the back padding in the dim dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the padding cell to load.
   * \param  index    The index of the padding cell in the dimension.
   * \param  dim      The dimension to load the padding in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto
  load_back(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    static_assert_iterator(it);
    *it = *it.offset(dim, index);
  }
};

} // namespace ripple

#endif // RIPPLE_PADDING_FO_EXTRAP_LOADER_HPP

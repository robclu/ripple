/**=--- ripple/padding/const_loader_.hpp ------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  const_loader.hpp
 * \brief This file defines an implementation of a padding loader which
 *        loads a contant into the boundary.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_CONST_LOADER_HPP
#define RIPPLE_PADDING_CONST_LOADER_HPP

#include "padding_loader.hpp"

namespace ripple {

/**
 * The ConstLoader type implements the padding loader interface to load the
 * boundary with a constant value.
 * \tparam T The type of the data for the boundary.
 */
template <typename T>
struct ConstLoader : public PaddingLoader<ConstLoader<T>> {
  T value = 0; //!< The value to load.

  /**
   * Constructor, sets  the constant value to load as the padding value.
   * \param v The value for the padding.
   */
  ripple_all ConstLoader(T v) noexcept : value(v) {}

  /**
   * Loads the front padding in the given dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the padding cell to load.
   * \param  index    The index of the padding cell in the dimension.
   * \param  dim      The dimension to load the padding in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_all constexpr auto
  load_front(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    *it = value;
  }

  /**
   * Loads the back padding in the given dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the padding cell to load.
   * \param  index    The index of the padding cell in the dimension.
   * \param  dim      The dimension to load the padding in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_all constexpr auto
  load_back(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    *it = value;
  }
};

} // namespace ripple

#endif // RIPPLE_PADDING_CONST_LOADER_HPP
//==--- ripple/core/boundary/const_loader.hpp -------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  const_loader.hpp
/// \brief This file defines an implementation of a boundary loader which
///        loads a contant into the boundary.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_CONST_LOADER_HPP
#define RIPPLE_BOUNDARY_CONST_LOADER_HPP

#include "boundary_loader.hpp"

namespace ripple {

/**
 * The  ConstLoader type implements the boundary loader interface to load the
 * boundary with a constant value.
 * \tparam T The type of the data for the boundary.
 */
template <typename T>
struct ConstLoader : public BoundaryLoader<ConstLoader<T>> {
  T value = 0; //!< The value to load.

  /**
   * Constructor, sets  the constant value to load as the boundary value.
   * \param v The value for the boundary.
   */
  ripple_host_device ConstLoader(T v) noexcept : value(v) {}

  /**
   * Loads the front boundary in the given dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the boundary cell to load.
   * \param  index    The index of the boundary cell in the dimension.
   * \param  dim      The dimension to load the boundary in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto
  load_front(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    static_assert_iterator(it);
    *it = value;
  }

  /**
   * Loads the back boundary in the given dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the boundary cell to load.
   * \param  index    The index of the boundary cell in the dimension.
   * \param  dim      The dimension to load the boundary in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto
  load_back(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    static_assert_iterator(it);
    *it = value;
  }
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_CONST_LOADER_HPP
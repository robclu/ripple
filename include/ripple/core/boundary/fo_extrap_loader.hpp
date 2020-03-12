//==--- ripple/core/boundary/copy_loader.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fo_extrap_loader.hpp
/// \brief This file defines an implementation of a boundary loader which
///        performs a first order (constant) extrapolation fromt the last cell
///        inside the domain to all boundary cells.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_FO_EXTRAP_LOADER_HPP
#define RIPPLE_BOUNDARY_FO_EXTRAP_LOADER_HPP

#include "boundary_loader.hpp"

namespace ripple {

/// The FOExtrapLoader is an implementation of an BoundaryLoader which copies
/// the data from the closest valid cell inside the domain to all boundary
/// cells.
struct FOExtrapLoader : public BoundaryLoader<FOExtrapLoader> {
  /// Loads the front boundary in the \p dim dimension, using the value of
  /// the \p index in the dimension to find the appropriate cell.
  /// \param  it       An iterator to the boundary cell to load.
  /// \param  index    The index of the boundary cell in the dimension.
  /// \param  dim      The dimension to load the boundary in.
  /// \tparam Iterator The type of the iterator.
  /// \tparam Dim      The type of the dimension specifier.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto load_front(
    Iterator&& it, int index, Dim&& dim
  ) const -> void {
    *it = *it.offset(dim, index);
  }

  /// Loads the back boundary in the \p dim dimension, using the value of
  /// the \p index in the dimension to find the appropriate cell.
  /// \param  it       An iterator to the boundary cell to load.
  /// \param  index    The index of the boundary cell in the dimension.
  /// \param  dim      The dimension to load the boundary in.
  /// \tparam Iterator The type of the iterator.
  /// \tparam Dim      The type of the dimension specifier.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto load_back(
    Iterator&& it, int index, Dim&& dim
  ) const -> void {
    *it = *it.offset(dim, index);
  }
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_FO_EXTRAP_LOADER_HPP

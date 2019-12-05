//==--- ripple/boundary/copy_loader.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  copy_loader.hpp
/// \brief This file defines an implementation of a boundary loader which copies
///        the boundary data from one iterator into another.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_COPY_LOADER_HPP
#define RIPPLE_BOUNDARY_COPY_LOADER_HPP

#include "internal_loader.hpp"

namespace ripple {

/// The CopyLoader is an implementation of an InternalLoader which simply copies
/// data from one iterator to another.
class CopyLoader : public InternalLoader<CopyLoader> {
  public:
  /// Loads the boundary in the \p dim dimension, by setting the data in the \p
  /// it_to iterator from the \p it_from iterator data.
  /// \param  it_from      An iterator to cell to load from.
  /// \param  it_to        An iterator to cell to load into.
  /// \tparam IteratorFrom The type of the from iterator.
  /// \tparam IteratorTo   The type of the to iterator.
  template <typename IteratorFrom, typename IteratorTo>
  ripple_host_device constexpr auto load(
    IteratorFrom&& it_from, IteratorTo&& it_to
  ) const -> void {
    *it_to = *it_from;
  } 
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_COPY_LOADER_HPP




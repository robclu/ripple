/**=--- ripple/padding/copy_loader.hpp --------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  copy_loader.hpp
 * \brief This file defines an implementation of an internal loader which
 *        copies the padding data from one iterator into another.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_COPY_LOADER_HPP
#define RIPPLE_PADDING_COPY_LOADER_HPP

#include "internal_loader.hpp"

namespace ripple {

/**
 * The CopyLoader is an implementation of an InternalLoader which simply copies
 * data from one iterator to another.
 */
struct CopyLoader : public InternalLoader<CopyLoader> {
  /**
   * Loads the padding in the given dimension by setting the data in the
   * to iterator from the from iterator data.
   * \param  from         An iterator to cell to load from.
   * \param  to           An iterator to cell to load into.
   * \tparam IteratorFrom The type of the from iterator.
   * \tparam IteratorTo   The type of the to iterator.
   */
  template <typename IteratorFrom, typename IteratorTo>
  ripple_all constexpr auto
  load(IteratorFrom&& from, IteratorTo&& to) const noexcept -> void {
    static_assert_iterator(to);
    static_assert_iterator(from);
    *to = *from;
  }
};

} // namespace ripple

#endif // RIPPLE_PADDING_COPY_LOADER_HPP

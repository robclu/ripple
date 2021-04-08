/**=--- ripple/padding/internal_loader.hpp ----------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  internal_loader.hpp
 * \brief This file defines an interface for inernal loading of data.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_INTERNAL_LOADER_HPP
#define RIPPLE_PADDING_INTERNAL_LOADER_HPP

#include <ripple/iterator/iterator_traits.hpp>
#include <ripple/utility/forward.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/**
 * The InternalLoader class defines an interface for internally loading data
 * for an iterator from another iterator, for cases such as shared and tiled
 * memory.
 * \tparam Impl The implementation of the loading interface.
 */
template <typename Impl>
class InternalLoader {
  /** Returns a const pointer to the implementation. */
  ripple_all constexpr auto impl() const -> const Impl* {
    return static_cast<const Impl*>(this);
  }

 protected:
  /**
   * Checks that the iterator is an iterator.
   * \tparam Iterator The iterator to check is an iterator.
   */
  template <typename Iterator>
  ripple_all auto
  static_assert_iterator(Iterator&&) const noexcept -> void {
    static_assert(
      is_iterator_v<Iterator>,
      "Boundary loader requires a parameter which is an iterator!");
  }

 public:
  /**
   * Loads the data in the to iterator using the data from the from
   * iterator.
   * \param  from         An iterator to cell to load from.
   * \param  to           An iterator to cell to load into.
   * \param  args         Additional arguments for the loading.
   * \tparam IteratorFrom The type of the from iterator.
   * \tparam IteratorTo   The type of the to iterator.
   * \tparam Args         The types of the additional arguments.
   */
  template <typename IteratorFrom, typename IteratorTo, typename... Args>
  ripple_all constexpr auto
  load(IteratorFrom&& from, IteratorTo&& to, Args&&... args) const noexcept
    -> void {
    impl()->load(
      ripple_forward(from), ripple_forward(to), ripple_forward(args)...);
  }
};

} // namespace ripple

#endif // RIPPLE_PADDING_INTERNAL_LOADER_HPP

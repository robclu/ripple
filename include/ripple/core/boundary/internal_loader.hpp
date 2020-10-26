//==--- ripple/core/boundary/inernal_loader.hpp ------------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                     Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  internal_loader.hpp
/// \brief This file defines an interface for inernal loading of data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_INTERNAL_LOADER_HPP
#define RIPPLE_BOUNDARY_INTERNAL_LOADER_HPP

#include <ripple/core/iterator/iterator_traits.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/**
 * The InternalLoader class defines an interface for internally loading data
 * for an iterator from another iterator, for cases such as shared and tiled
 * memory.
 * \tparam Impl The implementation of the loading interface.
 */
template <typename Impl>
class InternalLoader {
  /**
   * Returns a const pointer to the implementation.
   */
  ripple_host_device constexpr auto impl() const -> const Impl* {
    return static_cast<const Impl*>(this);
  }

 protected:
  /**
   * Checks that the iterator is an iterator.
   * \tparam Iterator The iterator to check is an iterator.
   */
  template <typename Iterator>
  ripple_host_device auto
  static_assert_iterator(Iterator&&) const noexcept -> void {
    static_assert(
      is_iterator_v<Iterator>,
      "Boundary loader requires a parameter which is an iterator!");
  }

 public:
  /**
   * Loads the data in the it_to iterator using the data from the it_from
   * iterator.
   * \param  it_from      An iterator to cell to load from.
   * \param  it_to        An iterator to cell to load into.
   * \param  args         Additional arguments for the loading.
   * \tparam IteratorFrom The type of the from iterator.
   * \tparam IteratorTo   The type of the to iterator.
   * \tparam Args         The types of the additional arguments.
   */
  template <typename IteratorFrom, typename IteratorTo, typename... Args>
  ripple_host_device constexpr auto
  load(IteratorFrom&& it_from, IteratorTo&& it_to, Args&&... args) const
    noexcept -> void {
    impl()->load(
      static_cast<IteratorFrom&&>(it_from),
      static_cast<IteratorTo&&>(it_to),
      static_cast<Args&&>(args)...);
  }
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_INTERNAL_LOADER_HPP

//==--- ripple/core/boundary/boundary_loader.hpp ----------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  boundary_loader.hpp
/// \brief This file defines an interface for boundary loading.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_BOUNDARY_LOADER_HPP
#define RIPPLE_BOUNDARY_BOUNDARY_LOADER_HPP

#include <ripple/core/iterator/iterator_traits.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/**
 * The BoundaryLoader class defines an interface for boundary loading.
 * \tparam Impl The implementation of the loading interface.
 */
template <typename Impl>
class BoundaryLoader {
  /**
   * Gets a const pointer to the implementation type.
   * \return A const pointer to the implementation.
   */
  ripple_host_device constexpr auto impl() const noexcept -> const Impl* {
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
   * Loads the front boundary in the \p dim dimension, using the value of
   * the \p index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the boundary cell to load.
   * \param  index    The index of the boundary cell in the dimension.
   * \param  dim      The dimension to load the boundary in.
   * \param  args     Additional arguments for the loading.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   * \tparam Args     The types of the additional arguments.
   */
  template <typename Iterator, typename Dim, typename... Args>
  ripple_host_device constexpr auto
  load_front(Iterator&& it, int index, Dim&& dim, Args&&... args) const noexcept
    -> void {
    impl()->load_front(
      static_cast<Iterator&&>(it),
      index,
      static_cast<Dim&&>(dim),
      static_cast<Args&&>(args)...);
  }

  /**
   * Loads the back boundary in the \p dim dimension, using the value of
   * the \p index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the boundary cell to load.
   * \param  index    The index of the boundary cell in the dimension.
   * \param  dim      The dimension to load the boundary in.
   * \param  args     Additional arguments for the loading.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   * \tparam Args     The types of the additional arguments.
   */
  template <typename Iterator, typename Dim, typename... Args>
  ripple_host_device constexpr auto
  load_back(Iterator&& it, int index, Dim&& dim, Args&&... args) const noexcept
    -> void {
    impl()->load_back(
      static_cast<Iterator&&>(it),
      index,
      static_cast<Dim&&>(dim),
      static_cast<Args&&>(args)...);
  }
};

/**
 * Determines if a type implements the loader interface.
 * \param T The type to check if is a boundary loader.
 */
template <typename T>
static constexpr bool is_loader_v =
  std::is_base_of_v<BoundaryLoader<std::decay_t<T>>, std::decay_t<T>>;

} // namespace ripple

#endif // RIPPLE_BOUNDARY_BOUNDARY_LOADER_HPP

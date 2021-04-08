/**=--- ripple/padding/padding_loader.hpp ------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  padding_loader.hpp
 * \brief This file defines an interface for boundary loading.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_PADDING_LOADER_HPP
#define RIPPLE_PADDING_PADDING_LOADER_HPP

#include <ripple/iterator/iterator_traits.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/**
 * The PaddingLoader class defines an interface for loading padding.
 * \tparam Impl The implementation of the loading interface.
 */
template <typename Impl>
class PaddingLoader {
  /**
   * Gets a const pointer to the implementation type.
   * \return A const pointer to the implementation.
   */
  ripple_all constexpr auto impl() const noexcept -> const Impl* {
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
      "Padding loader requires a parameter which is an iterator!");
  }

 public:
  /**
   * Loads the front padding in the dim dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the  padding cell to load.
   * \param  index    The index of the padding cell in the dimension.
   * \param  dim      The dimension to load the padding in.
   * \param  args     Additional arguments for the loading.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   * \tparam Args     The types of the additional arguments.
   */
  template <typename Iterator, typename Dim, typename... Args>
  ripple_all constexpr auto
  load_front(Iterator&& it, int index, Dim&& dim, Args&&... args) const noexcept
    -> void {
    impl()->load_front(
      ripple_forward(it), index, ripple_forward(dim), ripple_forward(args)...);
  }

  /**
   * Loads the back padding in the dim dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the padding cell to load.
   * \param  index    The index of the padding cell in the dimension.
   * \param  dim      The dimension to load the padding in.
   * \param  args     Additional arguments for the loading.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   * \tparam Args     The types of the additional arguments.
   */
  template <typename Iterator, typename Dim, typename... Args>
  ripple_all constexpr auto
  load_back(Iterator&& it, int index, Dim&& dim, Args&&... args) const noexcept
    -> void {
    impl()->load_back(
      ripple_forward(it), index, ripple_forward(dim), ripple_forward(args)...);
  }
};

/**
 * Determines if a type implements the loader interface.
 * \param T The type to check if is a boundary loader.
 */
template <typename T>
static constexpr bool is_loader_v =
  std::is_base_of_v<PaddingLoader<std::decay_t<T>>, std::decay_t<T>>;

} // namespace ripple

#endif // RIPPLE_PADDING_PADDING_LOADER_HPP

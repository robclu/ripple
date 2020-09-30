//==--- ripple/core/container/shared_wrapper.hpp ----------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  shared_wrapper.hpp
/// \brief This file defines a utility wrapper class for wrapping types to add
///        shared memory information.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_SHARED_WRAPPER_HPP
#define RIPPLE_CONTAINER_SHARED_WRAPPER_HPP

#include "block_traits.hpp"

namespace ripple {

/**
 * This is a wrapper type which stores a type which should be placed into
 * shared memory when passed to a kernel.
 *
 * This class simply wrapps the type to store in shared memory so that different
 * functions can be overloaded to enable or disable shared memory functionality.
 *
 * \tparam T The type to wrap for shared memory.
 */
template <typename T>
struct SharedWrapper {
  /** Defines a size for invalid padding. */
  static constexpr size_t invalid_padding = std::numeric_limits<size_t>::max();

  T      wrapped;                   //!< THe wrapped type for shared memory.
  size_t padding = invalid_padding; //!< The padding for the type.

  /**
   * Determines if the wrapped type is padded.
   * Returns true if the wraped type is padded.
   */
  auto padded() const noexcept -> bool {
    return padding != invalid_padding;
  }
};

/**
 * Overload of BlockEnabled traits for a SharedWrapper.
 *
 * This specialization exists so that the traits can be used for a wrapped type
 * as if the type was not wrapped.
 *
 * \tparam T The type being wrapped.
 */
template <typename T>
struct BlockEnabledTraits<SharedWrapper<T>> {
  /** Defines the number of dimensions for the type T if it's block enabled. */
  static constexpr size_t dimensions = block_enabled_traits_t<T>::dimensions;
};

/**
 * A traits class for shared wrapper types.
 * \tparam T The type to get the shared wrapper traits for.
 */
template <typename T>
struct SharedWrapperTraits {
  /** Defines the type being wrapped for shared memory. */
  using type = T;

  /** Returns that the type T is _not_ a SharedWrapper. */
  static constexpr bool is_shared_wrapper = false;
};

/**
 * Specialization of the shared wrapper traits for a type which is wrapped for
 * shared memory.
 * \tparam T The type being wrapped for shared memory.
 */
template <typename T>
struct SharedWrapperTraits<SharedWrapper<T>> {
  /** Defines the type being wrapped for shared memory. */
  using type = T;

  /** Returns that the type T _is_ a SharedWrapper. */
  static constexpr bool is_shared_wrapper = true;
};

/*==--- [aliases & constants] ----------------------------------------------==*/

/**
 * Returns the shared wrapper traits for T, removing any references for T.
 * \tparam T The type to get the shared wrapper traits for.
 */
template <typename T>
using shared_wrapper_traits_t = SharedWrapperTraits<remove_ref_t<T>>;

/**
 * Returns true if T is a shared wrapper.
 * \tparam T The type to check if is a shared wrapper.
 */
template <typename T>
static constexpr bool is_shared_wrapper_v =
  shared_wrapper_traits_t<T>::is_shared_wrapper;

/**
 * Defines a valid type if T __is__ a shared wrapper.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using shared_wrapper_enable_t = std::enable_if_t<is_shared_wrapper_v<T>, int>;

/**
 * Defines a valid type if T __is not__ a shared wrapper.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_shared_wrapper_enable_t =
  std::enable_if_t<!is_shared_wrapper_v<T>, int>;

/*==--- [functions] --------------------------------------------------------==*/

/**
 * Wraps a given type for shared mempry.
 * \tparam T The type to return a SharedWrapper for.
 * \return SharedWrapper for a type T
 */
template <typename T>
auto as_shared() noexcept -> SharedWrapper<T> {
  return SharedWrapper<T>{T{}};
}

/**
 * Wraps the argument for shared memory.
 * \param  t The object to wrap for shared memory.
 * \tparam T The type of the object to wrap.
 * \return A SharedWrapper which references the argument.
 */
template <typename T>
auto as_shared(T&& t) noexcept -> SharedWrapper<decltype(t)> {
  return SharedWrapper<decltype(t)>{static_cast<T&&>(t)};
}

/**
 * Wraps the argument for shared memory.
 * \param  t The object to wrap for shared memory.
 * \tparam T The type of the object to wrap.
 * \return A SharedWrapper which references the argument.
 */
template <typename T>
auto as_shared(T& t) noexcept -> SharedWrapper<T&> {
  return SharedWrapper<T&>{t};
}

/**
 * Wraps the argument for shared memory with the given amount of padding.
 * \param  t The object to wrap for shared memory.
 * \tparam T The type of the object.
 * \return A SharedWrapper which references the argument and has the given
 *         amount of padding.
 */
template <typename T>
auto as_shared(T& t, size_t padding) noexcept -> SharedWrapper<T&> {
  return SharedWrapper<T&>{t, padding};
}

} // namespace ripple

#endif // RIPPLE_CONTAINER_SHARED_WRAPPER_HPP
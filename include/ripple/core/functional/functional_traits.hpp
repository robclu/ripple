//==--- ripple/core/functional/functional_traits_.hpp ----------- -*- C++ -*-
//---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  functional_traits.hpp
/// \brief This file defines traits for functional functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_FUNCTIONAL_TRAITS_HPP
#define RIPPLE_FUNCTIONAL_FUNCTIONAL_TRAITS_HPP

#include <ripple/core/utility/portability.hpp>

namespace ripple {

/*==--- [forward declarations] ---------------------------------------------==*/

/**
 * The Invocable type defines an object which stores a functor which can be
 * invoked.
 *
 * \note This class always owns the functor.
 * \tparam Functor The type of the functor to invoke.
 */
template <typename Functor>
class Invocable;

/*==--- [traits helpers] ---------------------------------------------------==*/

namespace detail {

/**
 * Determines if the type T is Invocable or not. This specialization is for the
 * case that the type is not an invocable.
 *
 * \tparam T The type to determine if is invocable.
 */
template <typename T>
struct IsInvocable {
  /** Defines that the type T is not invocable. */
  static constexpr bool value = false;
};

/**
 * Specialization for an invocable type.
 * \tparam F The functor for the invocable.
 */
template <typename F>
struct IsInvocable<Invocable<F>> {
  /** Defines that the type is invocable. */
  static constexpr bool value = true;
};

} // namespace detail

/**
 * Returns true if T is an invocable type.
 * \tparam T The type to determine if is invocable.
 */
template <typename T>
static constexpr auto is_invocable_v =
  detail::IsInvocable<std::decay_t<T>>::value;

/**
 * Returns the type T as Invocable<std::decay_t<T>> if T is not already
 * invocable. It returns a decayed version of T because the Invocable type
 * always owns the callable.
 * \tparam T The type to check and potentially make invocable.
 */
template <typename T>
using make_invocable_t = std::
  conditional_t<is_invocable_v<std::decay_t<T>>, T, Invocable<std::decay_t<T>>>;

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_FUNCTIONAL_TRAITS_HPP

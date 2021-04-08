/**=--- ripple/algorithm/max_element.hpp ------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  max_element.hpp
 * \brief This file implements functionality to get the max element from a
 *        variadic pack.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ALGORITHM_MAX_ELEMENT_HPP
#define RIPPLE_ALGORITHM_MAX_ELEMENT_HPP

#include <ripple/utility/forward.hpp>
#include <ripple/utility/portability.hpp>
#include <algorithm>

namespace ripple {
namespace detail {

/**
 * Base case implementation for computing the max element. This just returns the
 * element since there is nothing to compare to.
 * \param  element The element to return.
 * \tparam T       The type of the element.
 * \return The element.
 */
template <typename T>
ripple_all constexpr auto
max_element_impl(T&& element) noexcept -> T&& {
  return ripple_forward(element);
}

/**
 * Implementation to compute the max of the current max and the next element and
 * then continue the computation.
 * \param current_max The current max element.
 * \param next        The next element in the pack.
 * \param rest        The remaining elements.
 * \tparam T           The type of the current max.
 * \tparam Next        The type of the next element.
 * \tparam Ts          The type of the rest of the elements.
 */
template <typename T, typename Next, typename... Ts>
ripple_all constexpr decltype(auto)
max_element_impl(T&& current_max, Next&& next, Ts&&... rest) noexcept {
  return max_element_impl(
    std::max(ripple_forward(current_max), ripple_forward(next)),
    ripple_forward(rest)...);
}

} // namespace detail

/**
 * Computes the max element from a variadic number of elements.
 *
 * \note This can be used at compile time to compute the max of a variadic pack.
 *
 * \param  first The first element in the pack.
 * \param  rest  The rest of the elements in the pack.
 * \tparam T     The type of the first element.
 * \tparam Ts    The types of the rest of the elements.
 * \return The max of the pack.
 */
template <typename T, typename... Ts>
ripple_all constexpr decltype(auto)
max_element(T&& first, Ts&&... rest) noexcept {
  return detail::max_element_impl(
    ripple_forward(first), ripple_forward(rest)...);
}

} // namespace ripple

#endif // RIPPLE_ALGORITHM_MAX_ELEMENT_HPP
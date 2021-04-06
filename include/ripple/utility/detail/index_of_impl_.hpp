/**=--- ripple/utility/detail/index_of_impl_.hpp ----------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  index_of_impl_.hpp
 * \brief This file defines an implemenatation for determining the index of a
 *        type in a parameter pack.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_UTILITY_DETAIL_INDEX_OF_IMPL__HPP
#define RIPPLE_UTILITY_DETAIL_INDEX_OF_IMPL__HPP

#include <type_traits>

namespace ripple::detail {

/**
 * Decalration of a class to determine the index of the type T in the Ts pack.
 * \tparam I  The current index of the search.
 * \tparam T  The type to find.
 * \tparam Ts The types to search through.
 */
template <int I, typename T, typename... Ts>
struct IndexOf;

/**
 * Decalration of a class to determine the index of the type T in the Ts pack.
 * Specialization for the general case.
 * \tparam I  The current index of the search.
 * \tparam T  The type to find.
 * \tparam U  The type to compare against T.
 * \tparam Ts The rest of the types to search through.
 */
template <int I, typename T, typename U, typename... Ts>
struct IndexOf<I, T, U, Ts...> {
  /** Returns the value I if T and U match, otherwise continues the search. */
  static constexpr auto value =
    std::is_same_v<T, U> ? I : IndexOf<I + 1, T, Ts...>::value;
};

/**
 * Base case when the search is exhausted, returns the original size of the
 * pack + 1.
 * \tparam I  The current index of the search.
 * \tparam T  The type to find.
 */
template <int I, typename T>
struct IndexOf<I, T> {
  /** Returns the exhausted search value. */
  static constexpr auto value = I;
};

} // namespace ripple::detail

#endif // RIPPLE_UTILITY_DETAIL_INDEX_OF_IMPL__HPP

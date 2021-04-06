/**=--- ripple/utiliy/type_traits.hpp ---------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  type_traits.hpp
 * \brief This file defines generic trait related functionality.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_UTILITY_TYPE_TRAITS_HPP
#define RIPPLE_UTILITY_TYPE_TRAITS_HPP

#include "detail/index_of_impl_.hpp"
#include "detail/function_traits_impl_.hpp"
#include "dim.hpp"
#include "number.hpp"
#include <type_traits>

namespace ripple {

/*==--- [traits] -----------------------------------------------------------==*/

/**
 * Returns true if the template parameter is a constexpr evaluatable number.
 * \tparam T The type to determine if is a constexpr number.
 */
template <typename T>
static constexpr bool is_cx_number_v = is_dimension_v<T> || is_number_v<T>;

/**
 * Returns true if all the types are arithmetic.
 *  \tparam Ts The types to check.
 */
template <typename... Ts>
static constexpr bool all_arithmetic_v =
  std::conjunction_v<std::is_arithmetic<std::decay_t<Ts>>...>;

/**
 * Returns true if any of the types are arithmetic.
 * \tparam Ts The types to check.
 */
template <typename... Ts>
static constexpr bool any_arithmetic_v =
  std::disjunction_v<std::is_arithmetic<std::decay_t<Ts>>...>;

/**
 * Returns true if all the types are the same.
 * \tparam T  The first type in the pack.
 * \tparam Ts The rest of the types to check.
 */
template <typename T, typename... Ts>
static constexpr bool all_same_v = std::conjunction_v<std::is_same<T, Ts>...>;

/**
 * Returns true if any of the types are the same as type T.
 * \tparam T  The first type in the pack to match against.
 * \tparam Ts The rest of the types to check.
 */
template <typename T, typename... Ts>
static constexpr bool any_same_v = std::disjunction_v<std::is_same<T, Ts>...>;

/**
 * Returns the index of the type T in the pack Ts. If T is not in the pack,
 * this will return sizeof...(Ts) + 1. It will also return false for types
 * with template types, unless they match exactly.
 * \tparam T  The type to search for the index of.
 * \tparam Ts The list of types to search in.
 */
template <typename T, typename... Ts>
static constexpr bool index_of_v = detail::IndexOf<0, T, Ts...>::value;

/*==--- [aliases] ----------------------------------------------------------==*/

/**
 * Defines a valid type if all the Ts are the same as type T.
 * \tparam T The type to base the enable on.
 * \tparam Ts The types to compare.
 */
template <typename T, typename... Ts>
using all_same_enable_t = std::enable_if_t<all_same_v<T, Ts...>, int>;

/**
 * Defines a valid type when the number of elements in the variadic pack
 * matches the size defined by Size, and all the types are arithmetic.
 * \tparam Size   The size that the pack must be.
 * \tparam Values The values in the pack.
 */
template <std::size_t Size, typename... Values>
using all_arithmetic_size_enable_t = std::
  enable_if_t<Size == sizeof...(Values) && all_arithmetic_v<Values...>, int>;

/**
 * Defines a valid type when the type T is not the same as the type U, when
 * they are both decayed.
 * \tparam T The first type for the comparison.
 * \tparam U The second type for the comparison.
 */
template <typename T, typename U>
using diff_enable_t =
  std::enable_if_t<!std::is_same_v<std::decay_t<T>, std::decay_t<U>>, int>;

/**
 * Defines a valid type when the number of elements in the variadic pack
 * matches the size defined by Size.
 * \tparam Size   The size that the pack must be.
 * \tparam Values The values in the pack.
 */
template <size_t Size, typename... Values>
using variadic_size_enable_t = std::enable_if_t<Size == sizeof...(Values), int>;

/**
 * Defines a valid type when the number of elements in the variadic pack
 * is greater than or equal to  the size defined by Size.
 * \tparam Size   The size that the pack must be.
 * \tparam Values The values in the pack.
 */
template <size_t Size, typename... Values>
using variadic_ge_enable_t = std::enable_if_t<(sizeof...(Values) >= Size), int>;

/**
 * Defines a valid type when the number of elements in the variadic pack
 * is less than the size defined by Size.
 * \tparam Size   The size that the pack must be.
 * \tparam Values The values in the pack.
 */
template <std::size_t Size, typename... Values>
using variadic_lt_enable_t = std::enable_if_t<(sizeof...(Values) < Size), int>;

/**
 * Defines a valid type when the type T is a pointer type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using pointer_enable_t = std::enable_if_t<std::is_pointer_v<T>, int>;

/**
 * Defines a valid type if Size is less than `ripple_max_unroll_depth`.
 * \tparam Size The size of the unrolling.
 */
template <size_t Size>
using unroll_enabled_t =
  std::enable_if_t<(Size < ripple_max_unroll_depth), int>;

/**
 * Defines a valid type if Size is more than or equal to
 * `ripple_max_unroll_depth`.
 * \tparam Size The size of the unrolling.
 */
template <size_t Size>
using unroll_disabled_t =
  std::enable_if_t<(Size >= ripple_max_unroll_depth), int>;

/**
 * Defines the type of the Nth element in the type list.
 * \tparam N   The index of the element to get.
 * \tparam Ts  The type list to get the type from.
 */
template <size_t N, typename... Ts>
using nth_element_t = typename detail::NthElement<N, Ts...>::type;

/*==--- [dimension enables] ------------------------------------------------==*/

/**
 * Defines a valid type for a single dimension (i.e Dims == 1).
 * \tparam Dims The number of dimensions to base the enable on.
 */
template <size_t Dims>
using dim_1d_enable_t = std::enable_if_t<Dims == 1, int>;

/**
 * Defines a valid type for any dimension execpt 1.
 * \tparam Dims The number of dimensions to base the enable on.
 */
template <size_t Dims>
using not_dim_1d_enable_t = std::enable_if_t<Dims != 1, int>;

/**
 * Defines a valid type for two dimensions (i.e Dims == 2).
 * \tparam Dims The number of dimensions to base the enable on.
 */
template <size_t Dims>
using dim_2d_enable_t = std::enable_if_t<Dims == 2, int>;

/**
 * Defines a valid type for any dimension except 2.
 * \tparam Dims The number of dimensions to base the enable on.
 */
template <std::size_t Dims>
using not_dim_2d_enable_t = std::enable_if_t<Dims != 2, int>;

/**
 * Defines a valid type for three dimensions (i.e Dims == 3).
 * \tparam Dims The number of dimensions to base the enable on.
 */
template <size_t Dims>
using dim_3d_enable_t = std::enable_if_t<Dims == 3, int>;

/**
 * Defines a valid type for any dimension except 3.
 * \tparam Dims The number of dimensions to base the enable on.
 */
template <size_t Dims>
using not_dim_3d_enable_t = std::enable_if_t<Dims != 3, int>;

/*==--- [function traits] --------------------------------------------------==*/

/**
 * Defines the traits for a function.
 * \tparam T The function type to get the traits for.
 */
template <typename T>
using function_traits_t = detail::FunctionTraits<T>;

} // namespace ripple

#endif // RIPPLE_UTILITY_TYPE_TRAITS_HPP

//==--- ripple/utility/type_traits.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Robb Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  type_traits.hpp
/// \brief This file defines generic trait related functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_TYPE_TRAITS_HPP
#define RIPPLE_UTILITY_TYPE_TRAITS_HPP

#include "detail/index_of_impl_.hpp"
#include "detail/nth_element_impl_.hpp"
#include "portability.hpp"
#include <type_traits>

namespace ripple {

//==--- [aliases] ----------------------------------------------------------==//

/// Returns true if all the types are arithmetic.
/// \tparam Ts The types to check.
template <typename... Ts>
static constexpr auto all_arithmetic_v = 
  std::conjunction_v<std::is_arithmetic<std::decay_t<Ts>>...>;

/// Returns true if any of the types are arithmetic.
/// \tparam Ts The types to check.
template <typename... Ts>
static constexpr auto any_arithmetic_v = 
  std::disjunction_v<std::is_arithmetic<std::decay_t<Ts>>...>;

/// Returns true if all the types are the same.
/// \tparam T  The first type in the pack.
/// \tparam Ts The rest of the types to check.
template <typename T, typename... Ts>
static constexpr auto all_same_v = 
  std::conjunction_v<std::is_same<T, Ts>...>;

/// Returns true if any of the types are the same as type T.
/// \tparam T  The first type in the pack to match against.
/// \tparam Ts The rest of the types to check.
template <typename T, typename... Ts>
static constexpr auto any_same_v = 
  std::disjunction_v<std::is_same<T, Ts>...>;

/// Returns true if the types T and U are the same, ignoring template parameters
/// if the types have template parameters. For example, for a type
///
/// ~~~
/// template <typename T> struct X {};
/// ~~~
/// 
/// with `T = X<int>, U = X<float>` this will return true.
///
/// This will work correctly for non-type parameters as well, and the number of
/// parameters can be anything.
///
/// The cases where this will not work are where a type mixes type and non-type
/// template parameters, or has non-type parameters which are different (e.g
/// size_t and int). These cases are rare, and the non-type parameters can
/// always be wrapped in the `Num` or `Int64` classes to convert them to types.
///
/// \tparam T The first type to compare.
/// \tparam U The second type to compare.
template <typename T, typename U>
static constexpr auto is_same_ignoring_templates_v =
  detail::IsSameIgnoringTemplates<T, U>::value;

/// Returns the index of the type T in the pack Ts. If T is not in the pack,
/// this will return sizeof...(Ts) + 1. It will also return false for types
/// with template types, unless they match exactly.
/// \tparam T  The type to search for the index of.
/// \tparam Ts The list of types to search in.
template <typename T, typename... Ts>
static constexpr auto index_of_v = detail::IndexOf<0, T, Ts...>::value;

/// Returns the index of the type T in the pack Ts. If T is not in the pack,
/// this will return sizeof...(Ts) + 1. 
///
/// This ignores the template types of both T and any of the Ts, and will match
/// on the outer type. For example, for some type
/// 
/// ~~~
/// template <typename T> struct X {};
/// ~~~
/// 
/// with `T = X<int>, Ts = <int, float, X<float>>`, this will return 2.
///
/// This will work correctly for non-type parameters, and the number of
/// parameters can be anything.
///
/// The cases where this will not work are where a type mixes type and non-type
/// template parameters, or has non-type parameters which are different (e.g
/// size_t and int). These cases are rare, and the non-type parameters can
/// always be wrapped in the `Num` or `Int64` classes to convert them to types.
///
/// \tparam T  The type to search for the index of.
/// \tparam Ts The list of types to search in.
template <typename T, typename... Ts>
static constexpr auto index_of_ignore_templates_v =
  detail::IndexOfIgnoreTemplates<0, T, Ts...>::value;

//==--- [traits] -----------------------------------------------------------==//

/// Defines a valid type when the number of elements in the variadic pack
/// matches the size defined by Size, and all the types are arithmetic.
/// \tparam Size   The size that the pack must be.
/// \tparam Values The values in the pack.
template <std::size_t Size, typename... Values>
using all_arithmetic_size_enable_t = std::enable_if_t<
  Size == sizeof...(Values) && all_arithmetic_v<Values...>, int
>;

/// Defines a valid type when the number of elements in the variadic pack
/// matches the size defined by Size.
/// \tparam Size   The size that the pack must be.
/// \tparam Values The values in the pack.
template <std::size_t Size, typename... Values>
using variadic_size_enable_t =
  std::enable_if_t<Size == sizeof...(Values), int>;

/// Defines a valid type when the type T is a pointer type.
/// \tparam T The type to base the enable on.
template <typename T>
using pointer_enable_t = std::enable_if_t<std::is_pointer_v<T>, int>;

/// Defines a valid type if Size is less than `ripple_max_unroll_depth`.
/// \tparam Size The size of the unrolling.
template <std::size_t Size>
using unroll_enabled_t = 
  std::enable_if_t<(Size < ripple_max_unroll_depth), int>;

/// Defines a valid type if Size is more than or equal to 
/// `ripple_max_unroll_depth`.
/// \tparam Size The size of the unrolling.
template <std::size_t Size>
using unroll_disabled_t = 
  std::enable_if_t<(Size >= ripple_max_unroll_depth), int>;

/// Defines the type of the Nth element in the type list.
/// \tparam N   The index of the element to get.
/// \tparam Ts  The type list to get the type from.
template <std::size_t N, typename... Ts>
using nth_element_t = typename detail::NthElement<N, Ts...>::type;

} // namespace ripple

#endif // RIPPLE_UTILITY_TYPE_TRAITS_HPP

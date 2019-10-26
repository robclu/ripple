//==--- streamline/utility/type_traits.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Streamline.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  type_traits.hpp
/// \brief This file defines generic trait related functionality.
//
//==------------------------------------------------------------------------==//

#ifndef STREAMLINE_UTILITY_TYPE_TRAITS_HPP
#define STREAMLINE_UTILITY_TYPE_TRAITS_HPP

#include "portability.hpp"
#include <type_traits>

//==--- [std wrappers] -----------------------------------------------------==//

namespace std    {
namespace detail {

/// Defines a default case for the conjuction.
template<class...> struct Conjunction : std::true_type {};

/// Specialization of the conjunction class for a single type.
/// \tparam Bool The bool type which defines the conjunction as true or false.
template<class Bool> struct Conjunction<Bool> : Bool {};

/// Specialization of the conjunction class for multiple types.
/// \tparam Bool  The first bool type.
/// \tparam Bools The rest of the bool types.
template<class Bool, class... Bools>
struct Conjunction<Bool, Bools...> 
: std::conditional_t<bool(Bool::value), Conjunction<Bools...>, Bool> {};

} // namespace detail

/// Forms the logical conjunction of the type traits B..., effectively
/// performing a logical AND on the sequence of traits.
///
/// \tparam Bools The types of the template bool parameters. Every template
///               argument Bi for which Bi::value is instantiated must be usable
///               as a base class and define member `value` that is convertible
///               to bool.
template <typename... Bools>
static constexpr bool conjunction_v = detail::Conjunction<Bools...>::value;

/// Wrapper around std::is_base_of for c++-14 compatibility.
/// \tparam Base The base type to use.
/// \tparam Impl The implementation to check if is a base of Base.
template <typename Base, typename Impl>
static constexpr bool is_base_of_v = 
  std::is_base_of<std::decay_t<Base>, std::decay_t<Impl>>::value;

/// Wrapper around std::is_convertible for c++14 compatibility.
/// \tparam From The type to convert from.
/// \tparam To   The type to convert to.
template <typename From, typename To>
static constexpr bool is_convertible_v =
  std::is_convertible<std::decay<From>, std::decay_t<To>>::value;

/// Wrapper around std::is_pointer for c++-14 compatibility.
/// \tparam T The type to determine if is a pointer.
template <typename T>
static constexpr bool is_pointer_v = std::is_pointer<std::decay_t<T>>::value;

/// Wrapper around std::is_same for c++-14 compatibility.
/// \tparam T1 The first type for the comparison.
/// \tparam T2 The second type for the comparison.
template <typename T1, typename T2>
static constexpr bool is_same_v =
  std::is_same<std::decay_t<T1>, std::decay_t<T2>>::value;

/// Wrapper around std::is_trivially_constructible for c++-14 compatibility.
/// \tparam T The type to determine if is trivially constructible.
template <typename T>
static constexpr bool is_trivially_constructible_v =
  std::is_trivially_constructible<std::decay_t<T>>::value;

} // namespace std

//==--- [streamline traits] ------------------------------------------------==//

namespace streamline {

/// Returns true if all the types are arithmetic.
/// \tparam Ts The types to check.
template <typename... Ts>
static constexpr auto all_arithmetic_v = 
  std::conjunction_v<std::is_arithmetic<Ts>...>;

/// Returns true if all the types are the same.
/// \tparam T  The first type in the pack.
/// \tparam Ts The rest of the types to check.
template <typename T, typename... Ts>
static constexpr auto all_same_v = 
  std::conjunction_v<std::is_same<T, Ts>...>;

/// Defines a valid type when the number of elements in the variadic pack
/// matches the size defined by Size, and all the types are arithmetic.
/// \tparam Size   The size that the pack must be.
/// \tparam Values The values in the pack.
template <std::size_t Size, typename... Values>
using arithmetic_size_enable_t = std::enable_if_t<
  Size == sizeof...(Values) && all_arithmetic_v<Values...>, int
>;

/// Defines a valid type when the number of elements in the variadic pack
/// matches the size defined by Size.
/// \tparam Size   The size that the pack must be.
/// \tparam Values The values in the pack.
template <std::size_t Size, typename... Values>
using variadic_size_enable_t =
  std::enable_if_t<Size == sizeof...(Values), int>;

/// Defines a valid type if Size is less than `streamline_max_unroll_depth`.
/// \tparam Size The size of the unrolling.
template <std::size_t Size>
using unroll_enabled_t = 
  std::enable_if_t<(Size < streamline_max_unroll_depth), int>;

/// Defines a valid type if Size is more than or equal to 
/// `streamline_max_unroll_depth`.
/// \tparam Size The size of the unrolling.
template <std::size_t Size>
using unroll_disabled_t = 
  std::enable_if_t<(Size >= streamline_max_unroll_depth), int>;

} // namespace streamline

#endif // STREAMLINE_UTILITY_TYPE_TRAITS_HPP

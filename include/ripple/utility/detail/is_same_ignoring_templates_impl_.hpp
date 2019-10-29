//==--- ../../detail/is_same_ignoring_templates_impl_.hpp -- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  is_same_ignoring_templates_impl_.hpp
/// \brief This file defines an implemenatation for determining if two types are
///        the same when template parameters are ignored.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_DETAIL_IS_SAME_IGNORING_TEMPLATES_IMPL__HPP
#define RIPPLE_UTILITY_DETAIL_IS_SAME_IGNORING_TEMPLATES_IMPL__HPP

#include <type_traits>

namespace ripple {
namespace detail {

/// Struct to determine if T is the same as U, ignoring template arguments.
/// \tparam T The first type for the comparison.
/// \tparam U The second type for the comparision.
template <typename T, typename U>
struct IsSameIgnoringTemplates {
  /// Returns true if T is the same as U.
    static constexpr auto value = std::is_same_v<T, U>;
};

/// Specialization for the case the where the type has different template
/// arguments and the arguments are types.
/// \tparam T  The first type for the comparison.
/// \tparam Ts The first set of template types for T.
/// \tparam Us The second set of template types for T.
template <template <typename...> typename T, typename... Ts, typename... Us>
struct IsSameIgnoringTemplates<T<Ts...>, T<Us...>> {
  /// Returns that the outer type is the same.
  static constexpr auto value = true;
};

/// Specialization for the case the where the type has different template
/// arguments and the arguments are non types.
/// \tparam T     The type for the comparison.
/// \tparam TType The type of the Ts arguments.
/// \tparam UType The type of the second set of arguments.
/// \tparam Ts    The first set of template types for T.
/// \tparam Us    The second set of template types for T.
template <
  typename                     TType,
  typename                     UType,
  template <TType...> typename T,
  TType...                     Ts,
  UType...                     Us
>
struct IsSameIgnoringTemplates<T<Ts...>, T<Us...>> {
  /// Returns true that the outer type is the same.
  static constexpr auto value = true;
};

}} // namespace ripple::detail

#endif // RIPPLE_UTILITY_DETAIL_IS_SAME_IGNORING_TEMPLATES_IMPL__HPP


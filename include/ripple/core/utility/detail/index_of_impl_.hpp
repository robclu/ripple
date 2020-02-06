//==--- ripple/core/utility/detail/index_of_impl_.hpp ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  index_of_impl_.hpp
/// \brief This file defines an implemenatation for determining the index of a
///        type in a parameter pack.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_DETAIL_INDEX_OF_IMPL__HPP
#define RIPPLE_UTILITY_DETAIL_INDEX_OF_IMPL__HPP

#include "is_same_ignoring_templates_impl_.hpp"

namespace ripple {
namespace detail {

//==--- [index of] ---------------------------------------------------------==//

/// Decalration of a class to determine the index of the type T in the Ts pack.
/// \tparam I  The current index of the search.
/// \tparam T  The type to find.
/// \tparam Ts The types to search through.
template <int I, typename T, typename... Ts>
struct IndexOf;

/// Decalration of a class to determine the index of the type T in the Ts pack.
/// Specialization for the general case.
/// \tparam I  The current index of the search.
/// \tparam T  The type to find.
/// \tparam U  The type to compare against T.
/// \tparam Ts The rest of the types to search through.
template <int I, typename T, typename U, typename... Ts>
struct IndexOf<I, T, U, Ts...> {
  /// Returns the value I if T and U match, otherwise continues the search.
  static constexpr auto value =
    std::is_same_v<T, U> ? I : IndexOf<I + 1, T, Ts...>::value ;
};

/// Base case when the search is exhausted, returns the original size of the
/// pack + 1.
/// \tparam I  The current index of the search.
/// \tparam T  The type to find.
template <int I, typename T>
struct IndexOf<I, T> {
  /// Returns the exhausted search value.
  static constexpr auto value = I;
};

//==--- [index of ignore templates] ----------------------------------------==//

/// Decalration of a class to determine the index of the type T in the Ts pack
/// where for all types which contain template parameters, the match on the
/// parameters is ignored.
/// \tparam I  The current index of the search.
/// \tparam T  The type to find.
/// \tparam Ts The types to search through.
template <int I, typename T, typename... Ts>
struct IndexOfIgnoreTemplates;

/// Decalration of a class to determine the index of the type T in the Ts pack
/// where for all types which contain template parameters, the match on the
/// template parameters is ignored.
/// Specialization for the general case.
/// \tparam I  The current index of the search.
/// \tparam T  The type to find.
/// \tparam U  The type to compare against T.
/// \tparam Ts The rest of the types to search through.
template <int I, typename T, typename U, typename... Ts>
struct IndexOfIgnoreTemplates<I, T, U, Ts...> {
  /// Returns the value I if T and U match, otherwise continues the search.
  static constexpr auto value =
    std::is_same_v<T, U> ? I : IndexOfIgnoreTemplates<I + 1, T, Ts...>::value;
};

/// Specialization for the case that the types to compare have template
/// parameters which are types.
/// \tparam I  The current index of the search.
/// \tparam T  The type to find.
/// \tparam U  The type to compare against T.
/// \tparam Ts The rest of the types to search through.
/// \tparam TTs The template types for type T.
/// \tparam UTs The tempalte types for type U.
template <
  int                             I,
  template <typename...> typename T,
  template <typename...> typename U,
  typename...                     Ts,
  typename...                     TTs,
  typename...                     UTs
>
struct IndexOfIgnoreTemplates<I, T<TTs...>, U<UTs...>, Ts...> {
  /// Returns the value I if T and U match, ignoring template parameters,
  /// otherwise continues the search.
  static constexpr auto value =
    std::is_same<T<TTs...>, U<UTs...>>::value || 
    IsSameIgnoringTemplates<T<TTs...>, U<UTs...>>::value
    ? I : IndexOfIgnoreTemplates<I + 1, T<TTs...>, Ts...>::value ;
};


/// Specialization for the case that the types to compare have template
/// parameters which are non typs.
/// \tparam I      The current index of the search.
/// \tparam TTType The type of T's template parameters.
/// \tparam UTType The type of U's template parameters.
/// \tparam T      The type to find.
/// \tparam U      The type to compare against T.
/// \tparam Ts     The rest of the types to search through.
/// \tparam TTs    The template types for type T.
/// \tparam UTs   The tempalte types for type U.
template <
  int                           I,
  typename                      TTType,
  typename                      UTType,
  template <TTType...> typename T,
  template <UTType...> typename U,
  typename...                   Ts,
  TTType...                     TTs,
  UTType...                     UTs
>
struct IndexOfIgnoreTemplates<I, T<TTs...>, U<UTs...>, Ts...> {
  /// Returns the value of the index I if type T and type U are the same
  /// ignoring the template arguments, otherwise continues the search.
  static constexpr auto value =
    IsSameIgnoringTemplates<T<TTs...>, U<UTs...>>::value
    ? I : IndexOfIgnoreTemplates<I + 1, T<TTs...>, Ts...>::value ;
};

/// Base case when the search is exhausted, returns the original size of the
/// pack + 1.
/// \tparam I  The current index of the search.
/// \tparam T  The type to find.
template <int I, typename T>
struct IndexOfIgnoreTemplates<I, T> {
  /// Returns the exhausted search value.
  static constexpr auto value = I;
};

}} // namespace ripple::detail

#endif // RIPPLE_UTILITY_DETAIL_INDEX_OF_IMPL__HPP

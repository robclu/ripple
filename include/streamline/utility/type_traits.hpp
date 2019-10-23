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

#include <type_traits>

//==--- [std wrappers] -----------------------------------------------------==//

namespace std {

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

/// Wrapper around std::is_same for c++-14 compatibility.
/// \tparam T1 The first type for the comparison.
/// \tparam T2 The second type for the comparison.
template <typename T1, typename T2>
static constexpr bool is_same_v =
  std::is_same<std::decay_t<T1>, std::decay_t<T2>>::value;

} // namespace std

//==--- [streamline traits] ------------------------------------------------==//

namespace streamline {

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

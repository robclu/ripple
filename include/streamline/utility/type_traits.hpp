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

namespace std {

/// Wrapper around std::is_base_of for c++-14 compatibility.
/// \tparam Base The base type to use.
/// \tparam Impl The implementation to check if is a base of Base.
template <typename Base, typename Impl>
static constexpr bool is_base_of_v = 
  std::is_base_of<std::decay_t<Base>, std::decay_t<Impl>>::value;

/// Wrapper around std::is_same for c++-14 compatibility.
/// \tparam T1 The first type for the comparison.
/// \tparam T2 The second type for the comparison.
template <typename T1, typename T2>
static constexpr bool is_same_v =
  std::is_same<std::decay_t<T1>, std::decay_t<T2>>::value;

} // namespace std

#endif // STREAMLINE_UTILITY_TYPE_TRAITS_HPP

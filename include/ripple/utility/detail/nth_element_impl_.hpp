//==--- ripple/utility/deatail/nth_element_impl_.hpp ------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Robb Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  nth_element_impl_.hpp
/// \brief This file defines an implementation for the Nth element in a type
///        list.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_DETAIL_NTH_ELEMENT_IMPL__HPP
#define RIPPLE_UTILITY_DETAIL_NTH_ELEMENT_IMPL__HPP

#include "../portability.hpp"
#include <type_traits>

namespace ripple {
namespace detail {

/// Defines a class which defines the type of the Nth element in the Ts pack.
/// \tparam N  The index of the type element to get.
/// \tparam Ts The types of the elements.
template <std::size_t N, typename... Ts>
struct NthElement {
  // If clang is being used, we can use the __much__ faster compiler intrinsic
  // to improve compile time, otherwise default to using tuple element.
#if ripple_clang
  /// Defines the type of the Ith element in the Ts pack.
  using type = __type_pack_element<N, Ts...>;
#else
  /// Defines the type of the Ith element in the Ts pack.
  using type = std::tuple_element_t<N, std::tuple<Ts...>>;
#endif
};

}} // namespace ripple::detail

#endif // RIPPLE_UTILITY_DETAIL_NTH_ELEMENT_IMPL__HPP


//==--- ripple/utility/detail/disjunction_impl_.hpp -------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  disjunction_impl_.hpp
/// \brief This file defines an implemenatation for compile time disjunction.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_DETAIL_DISJUNCTION_IMPL__HPP
#define RIPPLE_UTILITY_DETAIL_DISJUNCTION_IMPL__HPP

namespace std    {
namespace detail {

/// Defines a default case for the disjunction.
template<class...> struct Disjunction : std::false_type {};

/// Specialization of the disjunction class for a single type.
/// \tparam Bool The bool type which defines the disjunction as true or false.
template<class Bool> struct Disjunction<Bool> : Bool {};

/// Specialization of the disjunction class for multiple types.
/// \tparam Bool  The first bool type.
/// \tparam Bools The rest of the bool types.
template<class Bool, class... Bools>
struct Disjunction<Bool, Bools...> 
: std::conditional_t<bool(Bool::value), Bool, Disjunction<Bools...>> {};

}} // namespace std::detail

#endif // RIPPLE_UTILITY_DETAIL_DISJUNCTION_IMPL__HPP

//==--- ripple/utility/detail/conjunction_impl_.hpp -------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  conjunction_impl_.hpp
/// \brief This file defines an implemenatation for compile time conjunction.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_DETAIL_CONJUNCTION_IMPL__HPP
#define RIPPLE_UTILITY_DETAIL_CONJUNCTION_IMPL__HPP

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

}} // namespace std::detail

#endif // RIPPLE_UTILITY_DETAIL_CONJUNCTION_IMPL__HPP

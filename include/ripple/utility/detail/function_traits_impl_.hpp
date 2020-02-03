//==--- ripple/utility/detail/function_traits_.hpp --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  function_traits_.hpp
/// \brief This file defines the implementation of function traits
///        functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_DETAIL_FUNCTION_TRAITS_IMPL__HPP
#define RIPPLE_UTILITY_DETAIL_FUNCTION_TRAITS_IMPL__HPP

#include "nth_element_impl_.hpp"

namespace ripple::detail {

//==--- [function type] ----------------------------------------------------==//

/// Defines a class for types in a function.
/// \tparam Return     The return type of the function.
/// \tparam Class      The class type of the function.
/// \tparam IsConst    If the function is const.
/// \tparam IsVariadic If the function is variadic.
/// \tparam Args       The type of the arguments for the function.
template <
  typename    Return    ,
  typename    Class     ,
  bool        IsConst   ,
  bool        IsVariadic,
  typename... Args
>
struct FunctionTypes {
  /// Defines the arity of the function.
  static constexpr auto arity       = size_t{sizeof...(Args)};
  /// Defines if the function is const.
  static constexpr auto is_const    = IsConst;
  /// Defines if the function is variadic.
  static constexpr auto is_variadic = IsVariadic;

  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the return type of the function.
  using return_t = Return;
  /// Defins the class type of the function.
  using class_t  = Class;
  /// Defines the type of the Ith argument.
  /// \tparam I The index of the argument to get.
  template <size_t I>
  using arg_t = typename NthElement<I, Args...>::type;
};

//==--- [function traits] --------------------------------------------------==//

/// This type defines traits for a function. Note that this does not work for a
/// generic lambda, unless specific types are passes for the Ts types.
/// \tparam T  The type of the function.
/// \tparam Ts The types of the arguments to the function.
template <typename T, typename... Ts> struct FunctionTraits : 
FunctionTraits<decltype(&std::decay_t<T>::operator()(std::declval<Ts>()...))> 
{};

//==--- [function] ---------------------------------------------------------==//

/// Specialization of the function traits for a generic non-const function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args...)> : 
  FunctionTypes<R, void, false, false, Args...> {};

/// Specialization of the function traits for a generic const function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args...) const> : 
  FunctionTypes<R, void, true, false, Args...> {};

//==--- [pointer to function] ----------------------------------------------==//

/// Specialization of the function traits for a pointer to a non-const function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(*)(Args...)> : 
  FunctionTypes<R, void, false, false, Args...> {};

//==--- [reference to function] --------------------------------------------==//

/// Specialization of the function traits for a reference to a non-const
/// function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(&)(Args...)> : 
  FunctionTypes<R, void, false, false, Args...> {};

/// Specialization of the function traits for a reference to a non-const
/// function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args...)&> : 
  FunctionTypes<R, void, false, false, Args...> {};

/// Specialization of the function traits for a reference to const function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args...)const&> : 
  FunctionTypes<R, void, false, false, Args...> {};

/// Specialization of the function traits for an rvalue-reference to a non-const
/// function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args...)&&> : 
  FunctionTypes<R, void, false, false, Args...> {};

/// Specialization of the function traits for an rvalue-reference to const
/// function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args...)const&&> : 
  FunctionTypes<R, void, false, false, Args...> {};

//==--- [variadic function] ------------------------------------------------==//

/// Specialization of the function traits for a generic non-const variadic
/// function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args..., ...)> : 
  FunctionTypes<R, void, false, true, Args...> {};

/// Specialization of the function traits for a generic const variadic function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args..., ...) const> : 
  FunctionTypes<R, void, true, true, Args...> {};

//==--- [pointer to variadic function] -------------------------------------==//

/// Specialization of the function traits for a pointer to a non-const variadic
/// function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(*)(Args..., ...)> : 
  FunctionTypes<R, void, false, true, Args...> {};

//==--- [reference to variadic function] -----------------------------------==//

/// Specialization of the function traits for a reference to a non-const
/// variadic function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(&)(Args..., ...)> : 
  FunctionTypes<R, void, false, true, Args...> {};

/// Specialization of the function traits for a reference to a non-const
/// variadic function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args..., ...)&> : 
  FunctionTypes<R, void, false, true, Args...> {};

/// Specialization of the function traits for a reference to const variadic 
/// function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args..., ...)const&> : 
  FunctionTypes<R, void, false, true, Args...> {};

/// Specialization of the function traits for an rvalue-reference to a non-const
/// variadic function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args..., ...)&&> : 
  FunctionTypes<R, void, false, true, Args...> {};

/// Specialization of the function traits for an rvalue-reference to const
/// variafic function.
/// \tparam R     The return type of the function.
/// \tparam Args  The argument types of the function
template <typename R, typename... Args>
struct FunctionTraits<R(Args..., ...)const&&> : 
  FunctionTypes<R, void, false, true, Args...> {};

//==--- [pointer to class function] ----------------------------------------==//

/// Specialization of the function traits for a pointer to a non-const class
/// function.
/// \tparam R     The return type of the function.
/// \tparam C     The type of the class.
/// \tparam Args  The argument types of the function
template <typename R, typename C, typename... Args>
struct FunctionTraits<R(C::*)(Args...)> : 
  FunctionTypes<R, C, false, false, Args...> {};

/// Specialization of the function traits for a pointer to a const class
/// function.
/// \tparam R     The return type of the function.
/// \tparam C     The type of the class.
/// \tparam Args  The argument types of the function
template <typename R, typename C, typename... Args>
struct FunctionTraits<R(C::*)(Args...)const> : 
  FunctionTypes<R, C, true, false, Args...> {};

//==--- [pointer to class variadic function] -------------------------------==//

/// Specialization of the function traits for a pointer to a non-const class
/// variadic function.
/// \tparam R     The return type of the function.
/// \tparam C     The type of the class.
/// \tparam Args  The argument types of the function
template <typename R, typename C, typename... Args>
struct FunctionTraits<R(C::*)(Args..., ...)> : 
  FunctionTypes<R, C, false, true, Args...> {};

/// Specialization of the function traits for a pointer to a const class
/// variadic function.
/// \tparam R     The return type of the function.
/// \tparam C     The type of the class.
/// \tparam Args  The argument types of the function
template <typename R, typename C, typename... Args>
struct FunctionTraits<R(C::*)(Args..., ...)const> : 
  FunctionTypes<R, C, true, true, Args...> {};

} // namespace ripple::detail

#endif // RIPPLE_UTILITY_DETAIL_FUNCTION_TRAITS_IMPL__HPP

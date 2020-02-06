//==--- ripple/core/functional/functional_traits_.hpp ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  functional_traits.hpp
/// \brief This file defines traits for functional functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_FUNCTIONAL_TRAITS_HPP
#define RIPPLE_FUNCTIONAL_FUNCTIONAL_TRAITS_HPP

#include <ripple/core/utility/portability.hpp>

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// The Invocable type defines an object which stores a functor which can be
/// invoked, as well as some of the arguments with which to invoke it with.
///
/// This also allows some of the arguments not to be stored, but to be passed
/// when the invocable object is called. This is usefull for cases where the
/// invocable might be used with a static interface, where the implementation of
/// the interface is different each time the invocable us invoked:
///
/// ~~~{.cpp}
/// // Create the invocable with a const dt_val:
/// auto inv = make_invocable([] (auto iter, int dt) {
///   static_assert(is_iter_v<decltype(iter)>, "Not an iterator!");
///
///   *it += dt * (*it);
/// }, dt_val);
///
/// inv(iterator_to_use);
/// inv(different_iterator);
/// ~~~
///
/// This class always owns the functor and the arguments to the functor, it does
/// not reference either.
///
/// \tparam Functor The type of the functor to invoke.
/// \tparam Args    The type of the arguments for the functor.
template <typename Functor, typename... Args> class Invocable;

/// The Pipeline class stores a chain of operations which can be invoked. The
/// pipeline allows the ordering between the operations to be defined.
///
/// Pipelines should be created through the ```make_pipeline``` interface.
///
/// \tparam Ts The types of the invocable objects.
template <typename... Ts> class Pipeline;

//==--- [traits helpers] ---------------------------------------------------==//

namespace detail {

/// Determines if the type T is Invocable or not.
/// \tparam T The type to determine if is invocable.
template <typename T>
struct IsInvocable {
  /// Defines that the type T is not invocable.
  static constexpr bool value = false;
};

/// Specialization for an invocable type.
/// \tparam F    The functor for the invocable.
/// \tparam Args The args for the invocable.
template <typename F, typename... Args>
struct IsInvocable<Invocable<F, Args...>> {
  /// Defines that the type is invocable.
  static constexpr auto value = true;
};

/// Determines if the type T is a Pipeline or not.
/// \tparam T The type to determine if is a pipeline.
template <typename T>
struct IsPipeline {
  /// Defines that the type T is not a pipeline.
  static constexpr bool value = false;
};

/// Specialization for a pipeline.
/// \tparam Ops The operations in the pipeline.
template <typename... Ops>
struct IsPipeline<Pipeline<Ops...>> {
  /// Defines that the type is a pipeline.
  static constexpr auto value = true;
};

} // namespace detail

//==--- [constants] --------------------------------------------------------==//

/// Returns true if T is an invocable type.
/// \tparam T  The type to determine if is invocable.
/// \tparam DT The decayed type.
template <typename T, typename DT = std::decay_t<T>>
static constexpr auto is_invocable_v = detail::IsInvocable<DT>::value;

/// Returns true if T is a pipeline type.
/// \tparam T  The type to determine if is a pipeline.
/// \tparam DT The decayed type.
template <typename T, typename DT = std::decay_t<T>>
static constexpr auto is_pipeline_v = detail::IsPipeline<DT>::value;

//==--- [traits] -----------------------------------------------------------==//

/// Returns the type T as Invocable<T> if T is not already invocable.
/// \tparam T The type to check and potentially make invocable.
template <typename T>
using make_invocable_t = std::conditional_t<is_invocable_v<T>, T, Invocable<T>>;

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_FUNCTIONAL_TRAITS_HPP

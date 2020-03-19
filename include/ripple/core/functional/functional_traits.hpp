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
/// invoked. It's purpose is to be able to define function objects in a pipeline
/// which can then be invoked and synchronized. It can then be invoked using the
/// call operator as any callable.
///
///
/// ~~~{.cpp}
/// // Create the invocable with a const dt_val:
/// auto inv = make_invocable([] (auto iter, int dt) {
///   static_assert(is_iter_v<decltype(iter)>, "Not an iterator!");
///
///   *it += dt * (*it);
/// });
/// ~~~
///
/// This class always owns the functor.
/// \tparam Functor The type of the functor to invoke.
template <typename Functor> class Invocable;

/// The Pipeline class stores a chain of operations which can be invoked. The
/// purpose of the pipeline is to define a sequence of operations on a specific
/// set of data, where the data is synchronized across the block or the grid.
///
/// Pipelines should be created through the ```make_pipeline``` interface,
/// rather than through the pipeline class itself.
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
template <typename F>
struct IsInvocable<Invocable<F>> {
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
template <typename T>
static constexpr auto is_invocable_v = 
  detail::IsInvocable<std::decay_t<T>>::value;

/// Returns true if T is a pipeline type.
/// \tparam T  The type to determine if is a pipeline.
template <typename T>
static constexpr auto is_pipeline_v 
  = detail::IsPipeline<std::decay_t<T>>::value;

//==--- [traits] -----------------------------------------------------------==//

/// Returns the type T as Invocable<std::decay_t<T>> if T is not already 
/// invocable. It returns a decayed version of T because the Invocable type
/// always owns the callable.
/// \tparam T The type to check and potentially make invocable.
template <typename T>
using make_invocable_t = std::conditional_t<
  is_invocable_v<std::decay_t<T>>, T, Invocable<std::decay_t<T>>
>;

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_FUNCTIONAL_TRAITS_HPP

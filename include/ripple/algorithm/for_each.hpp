/**=--- ripple/algorithm/for_each_.hpp --------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  for_each.hpp
 * \brief This file implements a for each functionality on containers.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_CORE_ALGORITHM_FOR_EACH_HPP
#define RIPPLE_CORE_ALGORITHM_FOR_EACH_HPP

#include "unrolled_for.hpp"
#include <ripple/container/tuple.hpp>
#include <ripple/utility/forward.hpp>

namespace ripple {

/*==--- [tuple] ------------------------------------------------------------==*/

/**
 * Applies the functor to each of the elements in the tuple.
 *
 * \note This overload is for a non-const lvalue reference tuple.
 *
 * \param  tuple     The tuple to apply the functor to.
 * \param  functor   The functor to apply to each of the arguments.
 * \param  func_args Additional arguments for the functor.
 * \tparam Functor   The type of the functor.
 * \tparam Args      The type of the tuple arguments.
 * \tparam FuncArgs  Additional arguments for the functor.
 */
template <typename Functor, typename... Args, typename... FuncArgs>
ripple_all constexpr auto for_each(
  Tuple<Args...>& tuple, Functor&& functor, FuncArgs&&... func_args) noexcept
  -> void {
  constexpr size_t num_args = sizeof...(Args);
  unrolled_for<num_args>([&] ripple_all(auto i) -> void {
    functor(get<i>(tuple), ripple_forward(func_args)...);
  });
}

/**
 * Applies the functor to each of the elements in the tuple.
 *
 * \note This overload is for a a const reference tuple.
 *
 * \param  tuple     The tuple to apply the functor to.
 * \param  functor   The functor to apply to each of the arguments.
 * \param  func_args Additional arguments for the functor.
 * \tparam Functor   The type of the functor.
 * \tparam Args      The type of the tuple arguments.
 * \tparam FuncArgs  Additional arguments for the functor.
 */
template <typename Functor, typename... Args, typename... FuncArgs>
ripple_all constexpr auto for_each(
  const Tuple<Args...>& tuple,
  Functor&&             functor,
  FuncArgs&&... func_args) noexcept -> void {
  constexpr size_t num_args = sizeof...(Args);
  unrolled_for<num_args>([&] ripple_all(auto i) -> void {
    functor(get<i>(tuple), ripple_forward(func_args)...);
  });
}

/**
 * Applies the functor to each of the elements in the tuple.
 *
 * \note This overload is for an rvalue reference tuple.
 *
 * \param  tuple     The tuple to apply the functor to.
 * \param  functor   The functor to apply to each of the arguments.
 * \param  func_args Additional arguments for the functor.
 * \tparam Functor   The type of the functor.
 * \tparam Args      The type of the tuple arguments.
 * \tparam FuncArgs  Additional arguments for the functor.
 */
template <typename Functor, typename... Args, typename... FuncArgs>
ripple_all constexpr auto for_each(
  Tuple<Args...>&& tuple, Functor&& functor, FuncArgs&&... func_args) noexcept
  -> void {
  constexpr size_t num_args = sizeof...(Args);
  unrolled_for<num_args>([&] ripple_all(auto i) -> void {
    functor(ripple_move(get<i>(tuple)), ripple_forward(func_args)...);
  });
}

/*==--- [parameter pack] ---------------------------------------------------==*/

/**
 * Applies the functor to each of the args.
 *
 * \param  functor   The functor to apply to each of the arguments.
 * \param  args      The arguments to apply the functor to.
 * \tparam Functor   The type of the functor.
 * \tparam Args      The type of the arguments.
 */
template <typename Functor, typename... Args>
ripple_all constexpr auto
for_each(Functor&& functor, Args&&... args) -> void {
  using TupleType = Tuple<Args&&...>;
  for_each(TupleType{ripple_forward(args)...}, ripple_forward(functor));
}

} // namespace ripple

#endif // RIPPLE_CORE_ALGORITHM_FOR_EACH_HPP

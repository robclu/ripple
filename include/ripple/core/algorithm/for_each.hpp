//==--- ripple/core/algorithm/for_each.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  for_each.hpp
/// \brief This file implements a for each functionality on containers.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CORE_ALGORITHM_FOR_EACH_HPP
#define RIPPLE_CORE_ALGORITHM_FOR_EACH_HPP

#include "unrolled_for.hpp"
#include <ripple/core/container/tuple.hpp>

namespace ripple {

//==--- [tuple] ------------------------------------------------------------==//

/// Applies the \p functor to each of the elements in the \p tuple, when the
/// tuple is a non-const reference.
///
/// \param  tuple     The tuple to apply the functor to.
/// \param  functor   The functor to apply to each of the arguments.
/// \param  func_args Additional arguments for the functor.
/// \tparam Functor   The type of the functor.
/// \tparam Args      The type of the tuple arguments.
/// \tparam FuncArgs  Additional arguments for the functor.
template <typename Functor, typename... Args, typename... FuncArgs>
ripple_host_device constexpr auto for_each(
  Tuple<Args...>& tuple, Functor&& functor, FuncArgs&&... func_args
) -> void {
  constexpr auto num_args = sizeof...(Args);
  unrolled_for<num_args>([&] ripple_host_device (auto i) -> void {
    functor(get<i>(tuple), std::forward<FuncArgs>(func_args)...);
  });
}

/// Applies the \p functor to each of the elements in the \p tuple, when the
/// tuple is a const reference.
///
/// \param  tuple     The tuple to apply the functor to.
/// \param  functor   The functor to apply to each of the arguments.
/// \param  func_args Additional arguments for the functor.
/// \tparam Functor   The type of the functor.
/// \tparam Args      The type of the tuple arguments.
/// \tparam FuncArgs  Additional arguments for the functor.
template <typename Functor, typename... Args, typename... FuncArgs>
ripple_host_device constexpr auto for_each(
  const Tuple<Args...>& tuple, Functor&& functor, FuncArgs&&... func_args
) -> void {
  constexpr auto num_args = sizeof...(Args);
  unrolled_for<num_args>([&] ripple_host_device (auto i) -> void {
    functor(get<i>(tuple), std::forward<FuncArgs>(func_args)...);
  });
}

/// Applies the \p functor to each of the elements in the \p tuple, when the
/// tuple is a const reference.
///
/// \param  tuple     The tuple to apply the functor to.
/// \param  functor   The functor to apply to each of the arguments.
/// \param  func_args Additional arguments for the functor.
/// \tparam Functor   The type of the functor.
/// \tparam Args      The type of the tuple arguments.
/// \tparam FuncArgs  Additional arguments for the functor.
template <typename Functor, typename... Args, typename... FuncArgs>
ripple_host_device constexpr auto for_each(
  Tuple<Args...>&& tuple, Functor&& functor, FuncArgs&&... func_args
) -> void {
  constexpr auto num_args = sizeof...(Args);
  unrolled_for<num_args>([&] ripple_host_device (auto i) -> void {
    functor(std::move(get<i>(tuple)), std::forward<FuncArgs>(func_args)...);
  });
}

//==--- [parameter pack] ---------------------------------------------------==//

/// Applies the \p functor to each of the \p args.
/// \param  functor   The functor to apply to each of the arguments.
/// \param  args      The arguments to apply the functor to.
/// \tparam Functor   The type of the functor.
/// \tparam Args      The type of the arguments.
template <typename Functor, typename... Args>
ripple_host_device constexpr auto for_each(Functor&& functor, Args&&... args)
-> void {
  for_each(
    make_tuple(std::forward<Args>(args)...), std::forward<Functor>(functor)
  );
}

} // namespace ripple

#endif // RIPPLE_CORE_ALGORITHM_FOR_EACH_HPP

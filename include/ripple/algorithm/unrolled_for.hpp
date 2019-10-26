//==--- ripple/algorithm/unrolled-for.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  unrolled_for.hpp
/// \brief This file implements functionality for an unrolled for loop with a
///        compile-time size.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALGORITHM_UNROLLED_FOR_HPP
#define RIPPLE_ALGORITHM_UNROLLED_FOR_HPP

#include "detail/unrolled_for_impl_.hpp"
#include <ripple/utility/range.hpp>

namespace ripple {

/// Applies the functor Amount times and passes the index of the unrolling
/// as the first argument to the functor. The unrolling is performed at compile
/// time, so the number of iterations should not be too large. The index
/// parameter has a compile-time value, and can therefore be used in constexpr
/// contexts. For example:
/// 
/// ~~~cpp
/// auto tuple = std::make_tuple("string", 4, AType*());
/// unrolled_for<2>([&tuple] (auto i) {
///   do_something(get<i>(tuple), get<i + 1>(tuple)); 
/// });
/// ~~~
/// 
/// Which effectively will become:
/// 
/// ~~~cpp
/// do_something(get<0>(tuple), get<1>(tuple)); 
/// do_something(get<1>(tuple), get<2>(tuple)); 
/// ~~~
/// 
/// \note If it's required that the unrolling is bounded,  then use the
///       `unrolled_for_bounded`.
///
/// \param  functor   The functor to unroll.
/// \param  args      The arguments to the functor.
/// \tparam Amount    The amount of unrolling to do.
/// \tparam Functor   The type of the functor.
/// \tparam Args      The type of the functor arguments.
template <std::size_t Amount, typename Functor, typename... Args>
ripple_host_device constexpr inline auto unrolled_for(
  Functor&& functor, Args&&... args
) -> void {
  detail::Unroll<Amount> unrolled(
    std::forward<Functor>(functor), std::forward<Args>(args)...
  );
}

/// Applies the functor Amount times. However, this is a bounded version and
/// is safer than `unrolled_for` in that it will not unroll if the value of
/// Amount is larger than the value defined by `ripple_mac_unroll_depth` at
/// compile time. 
///
/// In the case that Amount is larger than `ripple_max_unroll_length` then
/// a normal loop is performed and the functor is invoked on each iteration. In
/// the case that `Amount <= ripple_max_unroll_depth`, then this behaves
/// exactly as `unrolled_for`. See `unrolled_for` for more details.
///
/// This version will not always used a constexpr index, and therefore the index
/// cannot be used in constexpr contexts.
///
/// This overload is only enabled when `Amount < ripple_max_unroll_depth`.
/// 
/// \param  functor   The functor to unroll.
/// \param  args      The arguments to the functor.
/// \tparam Amount    The amount of unrolling to do.
/// \tparam Functor   The type of the functor.
/// \tparam Args      The type of the functor arguments.
template <
  std::size_t Amount ,
  typename    Functor,
  typename... Args   ,
  unroll_enabled_t<Amount> = 0
>
ripple_host_device constexpr inline auto unrolled_for_bounded(
  Functor&& functor, Args&&... args
) -> void {
  unrolled_for<Amount>(
    std::forward<Functor>(functor),
    std::forward<Args>(args)...   
  );
}

/// Applies the functor Amount times. However, this is a bounded version and
/// is safer than `unrolled_for` in that it will not unroll if the value of
/// Amount is larger than the value defined by `ripple_mac_unroll_depth` at
/// compile time. 
///
/// In the case that Amount is larger than `ripple_max_unroll_length` then
/// a normal loop is performed and the functor is invoked on each iteration. In
/// the case that `Amount <= ripple_max_unroll_depth`, then this behaves
/// exactly as `unrolled_for`. See `unrolled_for` for more details.
///
/// This version will not always used a constexpr index, and therefore the index
/// cannot be used in constexpr contexts.
///
/// This overload is only enabled when `Amount >= ripple_max_unroll_depth`.
/// 
/// \param  functor   The functor to unroll.
/// \param  args      The arguments to the functor.
/// \tparam Amount    The amount of unrolling to do.
/// \tparam Functor   The type of the functor.
/// \tparam Args      The type of the functor arguments.
template <
  std::size_t Amount ,
  typename    Functor,
  typename... Args   ,
  unroll_disabled_t<Amount> = 0
>
ripple_host_device constexpr inline auto unrolled_for_bounded(
  Functor&& functor, Args&&... args
) -> void {
  for (const auto i : range(Amount)) {
    functor(i, std::forward<Args>(args)...);
  }
}

} // namespace ripple

#endif // RIPPLE_ALGORITHM_UNROLLED_FOR_HPP

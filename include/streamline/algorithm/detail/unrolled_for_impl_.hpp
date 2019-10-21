//==--- streamline/algorithm/detail/unrolled_for_impl_.hpp - -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  unrolled_for_impl_.hpp
/// \brief This file provides the implementation of the funtionality for compile
///        time based unrolling.
//
//==------------------------------------------------------------------------==//

#ifndef STREAMLINE_ALGORITHM_UNROLLED_FOR_IMPL_HPP
#define STREAMLINE_ALGORITHM_UNROLLED_FOR_IMPL_HPP

#include <streamline/utility/number.hpp>

namespace streamline {
namespace detail     {

/// The Unroll struct invokes a callable object N times, where the invokations
/// are unrolled at compile time.
/// \tparam Amount  The amount of unrolling to do.
template <std::size_t Amount>
struct Unroll : Unroll<Amount - 1> {
  /// Defines the value for the previous level.
  static constexpr std::size_t previous_level_v = Amount - 1;
  /// Defines the type of the case class which invokes at the previous level.
  using previous_level_t = Unroll<previous_level_v>;

  /// Passes the \p functor and \p args to the previous level to invoke, and
  /// then invokes at this level.
  /// \param[in]  functor   The functor to invoke.
  /// \param[in]  args      The arguments to pass to the functor.
  /// \tparam     Functor   The type of the functor to invoke.
  /// \tparam     Args      The type of the arguments to invoke with.
  template <typename Functor, typename... Args>
  stremaline_host_device Unroll(Functor&& functor, Args&&... args)
  : previous_level_t(
      std::forward<Functor>(functor), std::forward<Args>(args)... 
    ) {
    functor(Num<previous_level_v>(), std::forward<Args>(args)...);
  }
};

/// Specialization of the unrolling class to terminate the urolling at the
/// lowest level.
template <>
struct Unroll<1> {
  /// Invokes the functor with the given args.
  /// \param[in]  functor   The functor to invoke.
  /// \param[in]  args      The arguments to pass to the functor.
  /// \tparam     Functor   The type of the functor to invoke.
  /// \tparam     Args      The type of the arguments to invoke with.
  template <typename Functor, typename... Args>
  streamline_host_device Unroll(Functor&& functor, Args&&... args) {
    functor(Num<0>(), std::forward<Args>(args)...);
  }
};

/// Specialization of the unrolling class for the case that 0 unrolling is
/// specified. This specialization does nothing, it's defined for generic
/// code that may invoke it.
template <>
struct Unroll<0> {
  /// Does nothing.
  /// \param[in]  functor   Placeholder for a functor.
  /// \param[in]  args      Placeholder for the functor arguments.
  /// \tparam     Functor   The type of the functor.
  /// \tparam     Args      The type of the arguments.
  template <typename Functor, typename... Args>
  constexpr Unroll(Functor&&, Args&&...) {}
};

}} // namespace streamline::detail

#endif // STREAMLINE_ALGORITHM_UNROLLED_FOR_HPP

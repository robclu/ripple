/**=--- ripple/algorithm/detail/unrolled_for_impl_.hpp ----- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  unrolled_for_impl_.hpp
 * \brief This file provides the implementation of the funtionality for compile
 *        time based unrolling.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ALGORITHM_UNROLLED_FOR_IMPL_HPP
#define RIPPLE_ALGORITHM_UNROLLED_FOR_IMPL_HPP

#include <ripple/utility/forward.hpp>
#include <ripple/utility/number.hpp>

namespace ripple::detail {

/**
 * The Unroll struct invokes a callable object N times, where the invokations
 * are unrolled at compile time.
 *
 * This struct is an implementation detail for `unrolled_for`.
 *
 * \tparam Amount  The amount of unrolling to do.
 */
template <size_t Amount>
struct Unroll : public Unroll<(Amount <= 1 ? 0 : Amount - 1)> {
  /** Defines the value for the previous level. */
  static constexpr size_t previous_level = Amount <= 1 ? 0 : Amount - 1;

  /** Defines the type of the case class which invokes at the previous level. */
  using PreviousLevel = Unroll<previous_level>;

  /**
   * Passes the \p functor and \p args to the previous level to invoke, and
   * then invokes at this level.
   * \param  functor   The functor to invoke.
   * \param  args      The arguments to pass to the functor.
   * \tparam Functor   The type of the functor to invoke.
   * \tparam Args      The type of the arguments to invoke with.
   */
  template <typename Functor, typename... Args>
  ripple_all constexpr Unroll(
    Functor&& functor, Args&&... args) noexcept
  : PreviousLevel{ripple_forward(functor), ripple_forward(args)...} {
    functor(Num<previous_level>(), ripple_forward(args)...);
  }
};

/**
 * Specialization of the unrolling class to terminate the urolling at the
 * lowest level.
 *
 *  This struct is an implementation detail for `unrolled_for`.
 */
template <>
struct Unroll<1> {
  /**
   * Invokes the functor with the given args.
   * \param  functor   The functor to invoke.
   * \param  args      The arguments to pass to the functor.
   * \tparam Functor   The type of the functor to invoke.
   * \tparam Args      The type of the arguments to invoke with.
   */
  template <typename Functor, typename... Args>
  ripple_all constexpr Unroll(
    Functor&& functor, Args&&... args) noexcept {
    functor(Num<0>(), ripple_forward(args)...);
  }
};

/**
 * Specialization of the unrolling class for the case that 0 unrolling is
 * specified. This specialization does nothing, it's defined for generic
 * code that may invoke it.
 *
 * This struct is an implementation detail for `unrolled_for`.
 */
template <>
struct Unroll<0> {
  /**
   * Does nothing.
   * \param  functor   Placeholder for a functor.
   * \param  args      Placeholder for the functor arguments.
   * \tparam Functor   The type of the functor.
   * \tparam Args      The type of the arguments.
   */
  template <typename Functor, typename... Args>
  ripple_all constexpr Unroll(
    Functor&& functor, Args&&... args) noexcept {}
};

} // namespace ripple::detail

#endif // RIPPLE_ALGORITHM_UNROLLED_FOR_HPP

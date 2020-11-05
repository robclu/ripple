//==--- ripple/core/functional/invocable.hpp -------------------- -*- C++ -*-
//---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invocable.hpp
/// \brief This file implements functionality for an invocable object.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_INVOCABLE_HPP
#define RIPPLE_FUNCTIONAL_INVOCABLE_HPP

#include "functional_traits.hpp"
#include <ripple/core/container/tuple.hpp>
#include <ripple/core/utility/forward.hpp>
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

/**
 * The Invocable type defines an object which stores a functor which can be
 * invoked.
 *
 * \note This class always owns the functor object.
 *
 * \tparam FunctorType The type of the functor to invoke.
 */
template <typename FunctorType>
class Invocable {
  using Functor = FunctorType;

 public:
  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Takes a functor to store for the invocable.
   * \param functor The functor to store.
   * \tparam F The type of the functor.
   */
  template <typename F>
  ripple_host_device Invocable(F&& functor) noexcept
  : functor_{ripple_forward(functor)} {}

  /**
   * Takes a functor and copies it into this invocable.
   * \param functor The functor to store.
   */
  ripple_host_device Invocable(const Functor& functor) noexcept
  : functor_{functor} {}

  /**
   * Copy constructor which simply copies the functor from the other invocable
   * into this one.
   * \param other The other invocable object to copy.
   */
  ripple_host_device Invocable(const Invocable& other) noexcept
  : functor_{other.functor_} {}

  /**
   * Move constructor which moves the functor from the other invocable into this
   * one.
   * \param other The other invocable object to move.
   */
  ripple_host_device Invocable(Invocable&& other) noexcept
  : functor_{ripple_move(other._functor)} {}

  /**
   * Copy assignment to copy the invocable from the other invocable into this
   * one.
   * \param other The other invocable to copy from.
   * \return A reference to the modified invocable.
   */
  ripple_host_device auto
  operator=(const Invocable& other) noexcept -> Invocable& {
    functor_ = other.functor_;
    return *this;
  }

  /**
   * Move assignment to move the invocable from the other invocable into this
   * one.
   * \param other The other invocable to move into this one.
   * \return A reference to the modified invocable.
   */
  ripple_host_device auto operator=(Invocable&& other) noexcept -> Invocable& {
    if (&other != this) {
      functor_ = ripple_move(other.functor_);
    }
    return *this;
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Overload of the function call operator to invoke the invocable.
   *
   * This overload preserves the state of the invocable, passing the
   * arguments to the stored functor.
   *
   * \todo Add compile time check that the Args are all convertible to the
   *       type of the arguments for the functor.
   *
   * \param  args The arguments to invoke the functor with.
   * \tparam Args The types of the additional arguments.
   */
  template <typename... Args>
  ripple_host_device auto operator()(Args&&... args) const noexcept -> void {
    functor_(ripple_forward(args)...);
  }

  /**
   * Overload of the function call operator to invoke the invocable.
   *
   * This overload may not preserve the state of the invocable., and passes
   * the arguments to the stored functor.
   *
   * \todo Add compile time check that the Args are all convertible to the
   *       type of the arguments for the functor.
   *
   * \param  args The arguments to invoke the functor with.
   * \tparam Args The types of the additional arguments.
   */
  template <typename... Args>
  ripple_host_device auto operator()(Args&&... args) noexcept -> void {
    functor_(ripple_forward(args)...);
  }

 private:
  Functor functor_; //!< The functor for the invocable.
};

/*==--- [functions] -------------------------------------------------------==*/

/**
 * Creates an invocable object with the given functor.
 *
 * This function decays the functor type, because in order to be able to use
 * the invocable on both the host and the device, and across threads on the
 * host, it needs to be able to be moved easily as well as to not be a
 * reference to memory in the wrong address space (i.e reference to host
 * memory when on the device).
 *
 * \param  functor The functor which can be invoked.
 * \tparam Functor The type of the functor.
 * \return A newly created invocable.
 */
template <typename Functor>
ripple_host_device decltype(auto) make_invocable(Functor&& functor) noexcept {
  return Invocable<std::decay_t<Functor>>{ripple_forward(functor)};
}

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_INVOCABLE_HPP

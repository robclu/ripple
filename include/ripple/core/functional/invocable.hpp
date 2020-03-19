//==--- ripple/core/functional/invocable.hpp -------------------- -*- C++ -*- ---==//
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
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

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
template <typename Functor>
class Invocable {
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of the functor.
  using functor_t   = Functor;
  /// Defines the type of the invocable.
  using self_t      = Invocable<Functor>;

public:
  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines the arity of the functor.
  //static constexpr auto arity      = function_traits_t<functor_t>::arity;

  //==--- [construction] ---------------------------------------------------==//
  
  /// Takes a \p functor and either moves or copies the \p functor into this
  /// invocable.
  /// \param functor The functor to store.
  ripple_host_device Invocable(Functor functor) noexcept 
  : _functor(std::move(functor)) {}

  //==--- [copy & move construction] ---------------------------------------==//

  /// Copy constructor which simply copies the functor into this one.
  /// \param other The other invocable object to copy.
  ripple_host_device Invocable(const Invocable& other) noexcept 
  : _functor{other._functor} {}

  /// Move constructor which moves the functor from the \p other into this one.
  /// \param other The other invocable object to move.
  ripple_host_device Invocable(Invocable&& other) noexcept 
  : _functor{std::move(other._functor)} {}
  
  //==--- [copy & move assignment] -----------------------------------------==//
  
  /// Copy assignment to copy the invocable from the \p other invocable.
  /// \param other The other invocable to copy from.
  ripple_host_device auto operator=(const Invocable& other) noexcept
  -> self_t& {
    _functor = other._functor;
    return *this;
  }

  /// Move assignment to move the invocable from the \p other invocable.
  /// \param other The other invocable to move into this one.
  ripple_host_device auto operator=(Invocable&& other) noexcept
  -> self_t& {
    _functor = std::move(other._functor);
    return *this;
  }

  //==--- [interface] ------------------------------------------------------==//
 
  /// Overload of the call operator to invoke the invocable which is const. This
  /// overload preserves the state of the invocable. This passes the \p args
  /// arguments to the stored functor.
  ///
  /// \todo Add compile time check that the Args are all convertible to the type
  ///       of the arguments for the functor.
  ///
  /// \param  args The arguments to invoke the functor with.
  /// \tparam Args The types of the additional arguments. 
  template <typename... Args>
  ripple_host_device auto operator()(Args&&... args) const noexcept -> void {
    // assert_arg_type_match(
    //   std::make_index_sequence<sizeof...(Args)>(), 
    //   std::forward<Args>(args)...
    // );
    _functor(std::forward<Args>(args)...);
  }

  /// Overload of the call operator to invoke the invocable which is non const.
  /// This overload preserves the state of the invocable. This passes the
  /// \p args arguments to the stored functor.
  ///
  /// \todo Add compile time check that the Args are all convertible to the type
  ///       of the arguments for the functor.
  ///
  /// \param  args The arguments to invoke the functor with.
  /// \tparam Args The types of the additional arguments. 
  template <typename... Args>
  ripple_host_device auto operator()(Args&&... args) noexcept -> void {
    // assert_arg_type_match(
    //   std::make_index_sequence<sizeof...(Args)>(), 
    //   std::forward<Args>(args)...
    // );
    _functor(std::forward<Args>(args)...);
  }

 private:
  functor_t _functor; //!< The functor for the invocable.
};

//==--- [functions] --------------------------------------------------------==//

/// Creates an invocable object with a \p functor.
///
/// This function decays the functor type, because in order to be able to use
/// the invocable on botht the host and the device, and across threads on the
/// host, it needs to be able to be moved easily as well as to not be a
/// reference to memory in the wrong address space (i.e reference to host
/// memory when on the device).
///
/// \param  functor The functor which can be invoked.
/// \tparam Functor The type of the functor.
template <typename Functor>
ripple_host_device auto make_invocable(Functor&& functor) noexcept
-> Invocable<std::decay_t<Functor>> {
  return Invocable<std::decay_t<Functor>>{std::forward<Functor>(functor)};
}

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_INVOCABLE_HPP

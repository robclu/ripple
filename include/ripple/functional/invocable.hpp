//==--- ripple/functional/invocable.hpp -------------------- -*- C++ -*- ---==//
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

#include <ripple/container/tuple.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple {

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
template <typename Functor, typename... Args>
class Invocable {
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of the functor.
  using functor_t   = Functor;
  /// Defines a tuple of the arguments.
  using arg_tuple_t = Tuple<Args...>;
  /// Defines the type of the invocable.
  using self_t      = Invocable<Functor, Args...>;

public:
  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines the arity of the functor.
  //static constexpr auto arity      = function_traits_t<functor_t>::arity;
  /// Defines the number of fixed arguments.
  static constexpr auto fixed_args = sizeof...(Args);

  //==--- [construction] ---------------------------------------------------==//
  
  /// Takes a functor and and a pack or arguments and stores them as an
  /// invocable.
  ///
  /// This copies the functor and either copies or moves the arguments into the
  /// invocable to avoid the case that a reference to a function or one of the
  /// arguments was taken which goes out of scope.
  ///
  /// \param functor The functor to store.
  /// \param args    The arguments to store.
  ripple_host_device Invocable(const Functor& functor, Args&&... args) noexcept 
  : _args{arg_tuple_t{std::forward<Args>(args)...}}, _functor{functor} {} 

  /// Takes a functor and and a pack or arguments and stores them as an
  /// invocable.
  ///
  /// This moves both the functor and the arguments into the invocable to avoid
  /// the case that a reference to a function or one of the arguments was taken
  /// which goes out of scope.
  ///
  /// \param functor The functor to store.
  /// \param args    The arguments to store.
  ripple_host_device Invocable(Functor&& functor, Args&&... args) noexcept 
  : _args{arg_tuple_t{std::forward<Args>(args)...}}, 
    _functor{std::forward<Functor>(functor)} {}

  /// Takes a functor and and a tuple of arguments and stores them as an
  /// invocable.
  ///
  /// This moves both the functor and the arguments into the invocable to avoid
  /// the case that a reference to a function or one of the arguments was taken
  /// which goes out of scope.
  ///
  /// \param functor The functor to store.
  /// \param args    The arguments to store.
  ripple_host_device Invocable(Functor&& functor, arg_tuple_t&& args) noexcept
  : _args{std::move(args)}, 
    _functor{std::forward<Functor>(functor)} {}

  //==--- [copy & move construction] ---------------------------------------==//

  /// Copy constructor which simply copies the functor and the arguments.
  /// \param other The other invocable object to copy.
  ripple_host_device Invocable(const Invocable& other) noexcept 
  : _args{other._args}, _functor{other._functor} {}

  /// Move constructor which moves the functor and the arguments from \p other
  /// into this invocable.
  /// \param other The other invocable object to move from.
  ripple_host_device Invocable(Invocable&& other) noexcept 
  : _args{std::move(other._args)}, _functor{std::move(other._functor)} {}

  //==--- [copy & move assignment] -----------------------------------------==//
  
  /// Copy assignment to copy the invocable from the \p other invocable.
  /// \param other The other invocable to copy from.
  ripple_host_device auto operator=(const Invocable& other) noexcept
  -> self_t& {
    _args    = other._args;
    _functor = other._functor;
    return *this;
  }

  /// Move assignment to move the invocable from the \p other invocable.
  /// \param other The other invocable to move from.
  ripple_host_device auto operator=(Invocable&& other) noexcept -> self_t {
    _args    = std::move(other._args);
    _functor = std::move(other._functor);
    return *this;
  }

  //==--- [interface] ------------------------------------------------------==//
 
  /// Overload of the call operator to invoke the invocable which is const. This
  /// overload preserves the state of the invocable. This passes the \p ts
  /// arguents as the first arguments to the stored functor, and then forwards
  /// the arguments stored in the invocable.
  /// \tparam ts Additional arguments to invoke with.
  /// \tparam Ts The types of the additional arguments. 
  template <typename... Ts>
  ripple_host_device auto operator()(Ts&&... ts) const noexcept -> void {
    invoke_functor(
      std::make_index_sequence<fixed_args>(), std::forward<Ts>(ts)...
    );
  }

  /// Overload of the call operator to invoke the invocable which is non-const
  /// and which can therefore modify the state of the invocable my modifying any
  /// of the stored arguments. This passes the \p ts arguents as the first
  /// arguments to the stored functor, and then forwards the arguments stored
  /// in the invocable.
  /// \tparam ts Additional arguments to invoke with.
  /// \tparam Ts The types of the additional arguments. 
  template <typename... Ts>
  ripple_host_device auto operator()(Ts&&... ts) noexcept -> void {
    invoke_functor(
      std::make_index_sequence<fixed_args>(), std::forward<Ts>(ts)...
    );
  }
 private:
  arg_tuple_t _args;    //!< Arguments for the functor.
  functor_t   _functor; //!< The functor which can be invoked.

  /// Implemenatation of invocation which expands the stored arguments into the
  /// functor, and forwards the \p ts arguments to the functor.
  /// \param  indexer Object used to index the stored arguments.
  /// \param  ts      Additional arguments to the functor.
  /// \tparam I       The indices of the elements to expand.
  /// \tparam Ts      The types of additional arguments.
  template <size_t... I, typename... Ts>
  ripple_host_device auto invoke_functor(
    std::index_sequence<I...> indexer, Ts&&... ts
  ) const noexcept -> void {
    _functor(std::forward<Ts>(ts)..., get<I>(_args)...);
  }

  /// Implemenatation of invocation which expands the stored arguments into the
  /// functor, and forwards the \p ts arguments to the functor.
  /// \param  indexer Object used to index the stored arguments.
  /// \param  ts      Additional arguments to the functor.
  /// \tparam I       The indices of the elements to expand.
  /// \tparam Ts      The types of additional arguments.
  template <size_t... I, typename... Ts>
  ripple_host_device auto invoke_functor(
    std::index_sequence<I...> indexer, Ts&&... ts
  ) noexcept -> void {
    _functor(std::forward<Ts>(ts)..., get<I>(_args)...);
  }
};

//==--- [functions] --------------------------------------------------------==//

/// Creates an invocable object with a \p functor and its fixed \p args.
///
/// This function decays both the functor and the arguments, to force copies of
/// each, so that there are no dangling references. This allows the invocable to
/// be used in all scopes, transferred across threads, and on the device.
///
/// \param  functor The functor which can be invoked.
/// \param  args    The fixed arguments for the functor.
/// \tparam Functor The type of the functor.
/// \tparam Args    The type of the fixed arguments.
template <typename Functor, typename... Args>
ripple_host_device auto make_invocable(Functor&& functor, Args&&... args)
noexcept -> Invocable<std::decay_t<Functor>, std::decay_t<Args>...> {
  return Invocable<std::decay_t<Functor>, std::decay_t<Args>...>{
    static_cast<std::decay_t<Functor>>(functor), std::decay_t<Args>(args)...
  };
}

//==--- [traits] -----------------------------------------------------------==//

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

} // namespace detail

/// Returns true if T is an invocable type.
/// \tparam T The type to determine if is invocable.
template <typename T>
static constexpr auto is_invocable_v = 
  detail::IsInvocable<std::decay_t<T>>::value;

/// Returns the type T as Invocable<T> if T is not already invocable.
/// \tparam T The type to check and potentially make invocable.
template <typename T>
using make_invocable_t = std::conditional_t<is_invocable_v<T>, T, Invocable<T>>;

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_INVOCABLE_HPP

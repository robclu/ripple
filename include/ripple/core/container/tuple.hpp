//==--- ripple/core/container/tuple_.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tuple_.hpp
/// \brief This file implements a tuple class which can be used on both the host
///        and the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_TUPLE_HPP
#define RIPPLE_CONTAINER_TUPLE_HPP

#include "tuple_traits.hpp"
#include "detail/basic_tuple_.hpp"

namespace ripple {

//==--- [tuple implementation] ---------------------------------------------==//

/// Specialization for an empty Tuple.
template <> class Tuple<> {
  /// Defines the type of the storage.
  using storage_t = detail::BasicTuple<>;

 public:
  /// Intializes the Tuple with no elements.
  ripple_host_device constexpr Tuple() {}

  /// Retutns the size (0) of the tuple.
  ripple_host_device constexpr auto size() const -> size_t {
    return 0;
  }
};

/// Specialization for a non-empty Tuple.
/// \tparam Ts The types of the Tuple elements.
template <typename... Ts> 
class Tuple {
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of this tuple.
  using self_t = Tuple<Ts...>;
  //// Defines the type of the storage for the tuple.
  using storage_t = detail::BasicTuple<Ts...>;

  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines the number of elements in the tuple.
  static constexpr auto elements = size_t{sizeof...(Ts)};

  //=--- [construction] ----------------------------------------------------==//

  /// This overload of the constructor is called by the copy and move
  /// constructores to get the elements of the \p other Tuple and copy or move
  /// them into this Tuple.
  /// 
  /// \param  extractor Used to extract the elements out of \p other tuple.
  /// \param  other     The other tuple to copy or move.
  /// \tparam I         The indices for for extracting each element.
  /// \tparam T         The type of the other tuple.
  template <std::size_t... I, typename T>
  ripple_host_device constexpr explicit Tuple(
    std::index_sequence<I...> extractor, T&& other
  ) noexcept 
  : _storage{detail::get_impl<I>(std::forward<storage_t>(other.data()))...} {}

  storage_t _storage; //!< Storage for the tuple elements.

 public:
  //==--- [construction] ---------------------------------------------------==//
  
  /// Performs default initialization of the tuple types. 
  ripple_host_device constexpr Tuple() noexcept 
  : _storage{std::decay_t<Ts>()...} {}

  /// Intializes the tuple with a variadic list of const ref elements \p es.
  /// \param es The elements to store in the tuple.
  ripple_host_device constexpr Tuple(const Ts&... es) noexcept 
  : _storage{es...} {}

  /// Initializes the tuple with a variadic list of forwarding reference
  /// elements. This overload is only selcted if any of the types are not a
  /// tuple i.e this is not a copy or move constructor.
  ///  
  /// \param  es    The elements to store in the Tuple.
  /// \tparam Types The types of the elements.
  template <typename... Types, non_tuple_enable_t<Types...> = 0>
  ripple_host_device constexpr Tuple(Types&&... es) noexcept
  : _storage{std::forward<Types>(es)...} {}

  //==--- [move constructor] -----------------------------------------------==//

  /// Copy and move constructs the Tuple. This overload is only enable if the
  /// type T is a tuple.
  ///  
  /// \param  other The other tuple to copy or move.
  /// \tparam T     The type of the other tuple.
  template <typename T, tuple_enable_t<T> = 0>
  ripple_host_device constexpr explicit Tuple(T&& other) noexcept
  : Tuple{std::make_index_sequence<elements>{}, std::forward<T>(other)} {}

  //==--- [interface] ------------------------------------------------------==//
  
  /// Returns the number of elements in the tuple.
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return elements;
  }

  /// Returns a reference to the underlying storage container which holds the
  /// elements.
  ripple_host_device auto data() noexcept -> storage_t& { 
    return _storage; 
  }

  /// Returns a constant reference to the underlying storage container which
  /// holds the elements.
  ripple_host_device auto data() const noexcept -> const storage_t& { 
    return _storage;
  }
};

//==--- [get implemenation] ------------------------------------------------==//

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a const lvalue reference.
/// \param  tuple The Tuple to get the element from.
/// \tparam I     The index of the element to get from the Tuple.
/// \tparam Ts    The types of the Tuple elements.
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto get(
  const Tuple<Ts...>& tuple
) noexcept -> const std::decay_t<nth_element_t<I, Ts...>>& {
  return detail::get_impl<I>(tuple.data());
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a non-const reference type.
/// \param  tuple The Tuple to get the element from.
/// \tparam I     The index of the element to get from the Tuple.
/// \tparam Ts    The types of the Tuple elements.
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto get(Tuple<Ts...>& tuple) noexcept 
  -> std::decay_t<nth_element_t<I, Ts...>>& {
  return detail::get_impl<I>(tuple.data());
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a forwarding reference type.
/// \param  tuple The Tuple to get the element from.
/// \tparam I     The index of the element to get from the Tuple.
/// \tparam Ts    The types of the Tuple elements.
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto get(Tuple<Ts...>&& tuple) noexcept 
  -> std::decay_t<nth_element_t<I, Ts...>>&& {
  return detail::get_impl<I>(std::move(tuple.data()));
}

/// Defines a function to get a element from a Tuple. This overload is selected
/// when the \p tuple is a const orwarding reference type.
/// \param  tuple The Tuple to get the element from.
/// \tparam I     The index of the element to get from the Tuple.
/// \tparam Ts    The types of the Tuple elements.
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto get(
  const Tuple<Ts...>&& tuple
) noexcept -> const std::decay_t<nth_element_t<I, Ts...>>&& {
  return detail::get_impl<I>(std::move(tuple.data()));
}

//==--- [make tuple] -------------------------------------------------------==//

/// This makes a tuple, and is the interface through which tuples should be
/// created in almost all cases. Example usage is:
///
/// ~~~cpp
/// auto tuple = make_tuple(4, 3.5, "some value");  
/// ~~~
/// 
/// This imlementation decays the types, so it will not create refrence types,
/// i.e:
/// 
/// ~~~cpp
/// int x = 4, y = 5;
/// auto tup = make_tuple(x, y);
/// ~~~
/// 
/// will copy ``x`` and ``y`` and not create a tuple of references to the
/// variables. This is done so that in the default case the tuple can be used on
/// both the host and device, and passed to kernels.
///
/// However, if references are required, then that is an insance where explicit
/// creation of the tuple should be used:
///
/// ~~~cpp
/// int x = 0, y = 0.0f;
/// Tuple<int&, float&> tuple = make_tuple(x, y);
/// 
/// // Can modify x and y though tuple:
/// get<0>(tuple) = 4;
/// get<1>(tuple) = 3.5f;
/// 
/// // Can modify tuple values though x and y:
/// x = 0; y = 0.0f;
/// ~~~
/// 
/// \param  values The values to store in the tuple.
/// \tparam Ts     The types of the values for the tuple.
template <typename... Ts>
ripple_host_device constexpr inline auto make_tuple(Ts&&... values) noexcept {
  return Tuple<std::decay_t<Ts>...>(std::forward<std::decay_t<Ts>>(values)...);
}

} // namespace ripple:w


#endif // RIPPLE_CONTAINER_TUPLE_HPP


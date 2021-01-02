//==--- ripple/core/container/tuple.hpp -------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tuple.hpp
/// \brief This file implements a tuple class which can be used on both the host
///        and the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_TUPLE_HPP
#define RIPPLE_CONTAINER_TUPLE_HPP

#include "tuple_traits.hpp"
#include "detail/basic_tuple_.hpp"

namespace ripple {

/*==--- [tuple implementation] ---------------------------------------------==*/

/**
 * Specialization for an empty Tuple.
 */
template <>
class Tuple<> {
  /** Defines the type of the storage. */
  using Storage = detail::BasicTuple<>;

 public:
  /** Intializes the Tuple with no elements. */
  ripple_host_device constexpr Tuple() noexcept {}

  /**
   * Gets the size of the tuple.
   * \return The number of elements in the tuple.
   */
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return 0;
  }
};

/**
 * Specialization for a non-empty Tuple.
 * \tparam Ts The types of the Tuple elements.
 */
template <typename... Ts>
class Tuple {
  /** Defines the type of the storage for the tuple. */
  using Storage = detail::BasicTuple<Ts...>;

  /** Defines the number of elements in the tuple. */
  static constexpr size_t elements = size_t{sizeof...(Ts)};

  /*=--- [construction] ----------------------------------------------------==*/

  /**
   * This overload of the constructor is called by the copy and move
   * constructors to get the elements of the other tuple and copy or move them
   * into this tuple.
   *
   * \param  other     The other tuple to copy or move.
   * \param  extractor Used to extract the elements out of \p other tuple.
   * \tparam T         The type of the other tuple.
   * \tparam I         The indices for for extracting each element.
   */
  template <typename T, size_t... I>
  ripple_host_device constexpr explicit Tuple(
    T&& other, std::index_sequence<I...> extractor) noexcept
  : storage_{detail::get_impl<I>(static_cast<Storage&&>(other.data()))...} {}

  Storage storage_; //!< Storage for the tuple elements.

 public:
  /**
   * Performs default initialization of the tuple types.
   */
  ripple_host_device constexpr Tuple() noexcept
  : storage_{std::decay_t<Ts>()...} {}

  /**
   * Intializes the tuple with a variadic list of const ref elements.
   * \param es The elements to store in the tuple.
   */
  ripple_host_device constexpr Tuple(const Ts&... es) noexcept
  : storage_{es...} {}

  /**
   * Initializes the tuple with a variadic list of forwarding reference
   * lements. This overload is only selcted if any of the types are not a
   * tuple i.e this is not a copy or move constructor.
   *
   *
   * \param  es    The elements to store in the Tuple.
   * \tparam Types The types of the elements.
   */
  template <typename... Types, non_tuple_enable_t<Types...> = 0>
  ripple_host_device constexpr Tuple(Types&&... es) noexcept
  : storage_{static_cast<Types&&>(es)...} {}

  /*==--- [move constructor] -----------------------------------------------==*/

  /**
   * Copy or move constructs the Tuple, depending on whether T is a rvalue or
   * lvalue reference.
   *
   * \note This overload is only enable if the type T is a tuple.
   *
   * \param  other The other tuple to copy or move.
   * \tparam T     The type of the other tuple.
   */
  template <typename T, tuple_enable_t<T> = 0>
  ripple_host_device constexpr explicit Tuple(T&& other) noexcept
  : Tuple{static_cast<Tuple&&>(other), std::make_index_sequence<elements>{}} {}

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets the number of elements in the tuple.
   * \return The number of elements in the tuple.
   */
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return elements;
  }

  /**
   * Gets a reference to the underlying storage container which holds the
   * elements.
   * \return A reference to the data for the tuple.
   */
  ripple_host_device auto data() noexcept -> Storage& {
    return storage_;
  }

  /**
   * Gets a constant reference to the underlying storage container which
   * holds the elements.
   * \return A const reference to the data for the tuple.
   */
  ripple_host_device auto data() const noexcept -> const Storage& {
    return storage_;
  }
};

/*==--- [get implemenation] ------------------------------------------------==*/

/**
 * Defines a function to get an element from a Tuple.
 *
 * \note This overload is selected when the tuple is a const lvalue reference.
 *
 * \param  tuple The Tuple to get the element from.
 * \tparam I     The index of the element to get from the Tuple.
 * \tparam Ts    The types of the Tuple elements.
 * \return A const reference to the Ith element.
 * */
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto get(const Tuple<Ts...>& tuple) noexcept
  -> const std::decay_t<nth_element_t<I, Ts...>>& {
  return detail::get_impl<I>(tuple.data());
}

/**
 * Defines a function to get an element from a Tuple.
 *
 * \note This overload is selected when the tuple is a non-const reference
 *       type.
 *
 * \param  tuple The Tuple to get the element from.
 * \tparam I     The index of the element to get from the Tuple.
 * \tparam Ts    The types of the Tuple elements.
 * \return A reference to the Ith element.
 */
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto get(Tuple<Ts...>& tuple) noexcept
  -> std::remove_reference_t<nth_element_t<I, Ts...>>& {
  return detail::get_impl<I>(tuple.data());
}

/**
 * Defines a function to get an element from a Tuple.
 *
 * \note This overload is selected  when the tuple is a forwarding reference
 *       type.
 *
 * \param  tuple The Tuple to get the element from.
 * \tparam I     The index of the element to get from the Tuple.
 * \tparam Ts    The types of the Tuple elements.
 * \return An rvalue reference to to the element.
 */
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto get(Tuple<Ts...>&& tuple) noexcept
  -> std::remove_reference_t<nth_element_t<I, Ts...>>&& {
  using DataType = decltype(tuple.data());
  return detail::get_impl<I>(
    static_cast<std::remove_reference_t<DataType>&&>(tuple.data()));
}

/**
 * Defines a function to get a element from a Tuple.
 *
 * \note This overload is selected when the tuple is a const orwarding reference
 *       type.
 *
 * \param  tuple The Tuple to get the element from.
 * \tparam I     The index of the element to get from the Tuple.
 * \tparam Ts    The types of the Tuple elements.
 * \return A const rvalue reference to the element.
 */
template <size_t I, typename... Ts>
ripple_host_device constexpr inline auto
get(const Tuple<Ts...>&& tuple) noexcept
  -> const std::decay_t<nth_element_t<I, Ts...>>&& {
  using DataType = decltype(tuple.data());
  return detail::get_impl<I>(
    static_cast<const std::remove_reference_t<DataType>&&>(tuple.data()));
}

/*==--- [make tuple] -------------------------------------------------------==*/

/**
 * This makes a tuple, and is the interface through which tuples should be
 * created in almost all cases. Example usage is:
 *
 * ~~~cpp
 * auto tuple = make_tuple(4, 3.5, "some value");
 * ~~~
 *
 * This imlementation decays the types, so it will not create refrence types,
 * i.e:
 *
 * ~~~cpp
 * int x = 4, y = 5;
 * auto tup = make_tuple(x, y);
 * ~~~
 *
 * will copy ``x`` and ``y`` and not create a tuple of references to the
 * variables. This is done so that in the default case the tuple can be used on
 * both the host and device, and passed to kernels.
 *
 * \note If references are required, then explicit creation of the tuple should
 *       be used:
 *
 *
 * ~~~cpp
 * int x = 0, y = 0.0f;
 * Tuple<int&, float&> tuple{x, y};
 *
 * // Can modify x and y though tuple:
 * get<0>(tuple) = 4;
 * get<1>(tuple) = 3.5f;
 *
 * // Can modify tuple values though x and y:
 * x = 0; y = 0.0f;
 * ~~~
 *
 * \param  values The values to store in the tuple.
 * \tparam Ts     The types of the values for the tuple.
 * \return A tuple containing the values.
 */
template <typename... Ts>
ripple_host_device constexpr inline auto
make_tuple(Ts&&... values) noexcept -> Tuple<std::decay_t<Ts>...> {
  return Tuple<std::decay_t<Ts>...>{static_cast<Ts&&>(values)...};
}

} // namespace ripple

#endif // RIPPLE_CONTAINER_TUPLE_HPP

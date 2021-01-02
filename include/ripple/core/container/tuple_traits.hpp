//==--- ripple/core/container/tuple_traits.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tuple_traits.hpp
/// \brief This file implements defines tuple related traits.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_TUPLE_TRAITS_HPP
#define RIPPLE_CONTAINER_TUPLE_TRAITS_HPP

#include "detail/basic_tuple_.hpp"
#include "../utility/type_traits.hpp"

namespace ripple {

/*==--- [forward declarations] ---------------------------------------------==*/

/**
 * The Tuple type holds a fixed-size number of heterogenous types.
 *
 * This implementation of a tuple is similar to std::tuple, but allows the tuple
 * to be used on both the host and the device. We could use thrust::tuple,
 * however, it only supports 9 types (rather than the compiler's recursion
 * depth limit in the case of this implementation), since it doesn't use
 * variadic tempates, which is also limiting.
 *
 * It also would then add an external dependency.
 *
 * Having a separate implementation also allows for more functionality to be
 * added where required.
 *
 * The interface through which a tuple should be created is the make_tuple()
 * function, i.e:
 *
 * ~~~cpp
 * auto tuple = make_tuple(4, 3.5, "some value");
 * ~~~
 *
 * unless the tuple needs to store reference types. See the documentation for
 * make_tuple for an example.
 *
 * \note If the tuple stores reference type, and is created on the host, then
 *       it will not work correctly on the device.
 *
 *
 * \tparam Ts The types of the elements in the tuple.
 */
template <typename... Types>
class Tuple;

/*==--- [traits] -----------------------------------------------------------==*/

namespace detail {

/**
 * This type defines tuple traits for a type T which is __not__ a tuple.
 * \tparam T The type to get the tuple traits for.
 */
template <typename T>
struct TupleTraits {
  /** Defines the size of the tuple. */
  static constexpr size_t size = 0;
};

/**
 * This type defines tuple traits for a tuple with Ts types.
 * \tparam Ts The types stored by the tuple.
 */
template <typename... Ts>
struct TupleTraits<Tuple<Ts...>> {
  /** Defines the size of the tuple. */
  static constexpr size_t size = sizeof...(Ts);
};

/**
 * This type determines if a type T is a Tuple. This is the default
 * implementation for the case that the type T is not a tuple.
 * \tparam T The type to check against tuple.
 */
template <typename T>
struct IsTuple {
  /** Returns that the type T is not a tuple. */
  static constexpr bool value = false;
};

/**
 * Specialization for the case that the type is a tuple.
 * \tparam Ts The types the tuple stores.
 */
template <typename... Ts>
struct IsTuple<Tuple<Ts...>> {
  /** Returns that the type is a tuple. */
  static constexpr bool value = true;
};

/**
 * The TupleElement class get the type of the element at index I in a tuple.
 * \tparam I The index of the element to get type type of.
 * \tparam T The type of the tuple.
 */
template <size_t I, typename T>
struct TupleElement {
  /** Defines the type of the data for the tuple type T. */
  using DataType = decltype(std::declval<T>().data());

  /** Defines the type of the element at index Idx. */
  using Type =
    decltype(type_extractor<I>(static_cast<std::remove_reference_t<DataType>&&>(
      std::declval<T>().data())));
};

/**
 * Returns the type of a tuple element for a tuple with no elements,
 * \tparam I The index of the element to get the type of.
 */
template <size_t I>
struct TupleElement<I, Tuple<>> {
  /** Defines the type of the element. */
  using Type = void;
};

} // namespace detail

/**
 *  Defines the type of tuple traits for a type T.
 * \tparam T The type to get the tuple traits for.
 */
template <typename T>
using tuple_traits_t = detail::TupleTraits<std::decay_t<T>>;

/**
 * Defines the type of the element at position I in a tuple T.
 * \tparam I The index of the element to get the type for.
 * \tparam T The type of the tuple to get the element type from.
 */
template <size_t I, typename T>
using tuple_element_t = typename detail::TupleElement<I, T>::Type;

/**
 * Returns true if a decayed T is a tuple, and false otherwise.
 * \tparam T The type to determine if is a tuple.
 */
template <typename T>
static constexpr bool is_tuple_v = detail::IsTuple<std::decay_t<T>>::value;

/**
 * Returns the size of the tuple, if T is a tuple, otherwise returns 0.
 * \tparam T The type to get the tuple size of.
 */
template <typename T>
static constexpr size_t tuple_size_v = tuple_traits_t<T>::size;

/*==--- [enables] ----------------------------------------------------------==*/

/**
 * Defines a valid type if the first element in the variadic pack is a tuple,
 * otherwise does not define a valid type.
 * \tparam T  The type to base the enable on.
 * \tparam Ts The rest of the types which do not effect the enable.
 */
template <typename... Ts>
using tuple_enable_t =
  std::enable_if_t<is_tuple_v<nth_element_t<0, Ts...>>, int>;

/**
 * Defines a valid type if the first type in the variadic pack is *not* a
 * tuple, otherwise does not define a valid type.
 * \tparam T  The type to base the enable on.
 * \tparam Ts The rest of the types which do not effect the enable.
 */
template <typename... Ts>
using non_tuple_enable_t =
  std::enable_if_t<!is_tuple_v<nth_element_t<0, Ts...>>, int>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_TUPLE_HPP

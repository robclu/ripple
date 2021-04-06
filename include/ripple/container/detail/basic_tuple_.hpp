/**=--- ripple/container/detail/basic_tuple_.hpp ----------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  basic_tuple_.hpp
 * \brief This file implements the lowest level functionality for a tuple
 *        class.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_CONTAINER_DETAIL_BASIC_TUPLE__HPP
#define RIPPLE_CONTAINER_DETAIL_BASIC_TUPLE__HPP

#include <ripple/utility/portability.hpp>
#include <utility>

namespace ripple::detail {

/*==--- [element] ----------------------------------------------------------==*/

/**
 * Defines a struct to hold an element at a specific index in a container.
 * \tparam Index The index of the element being held.
 * \tparam T     The type of the element.
 */
template <size_t Index, typename T>
struct Element {
  /**
   * Default constructor which just initializes the stored type.
   */
  constexpr Element() = default;

  /**
   * Constructor to set the value of the element.
   * \param  element The element to use to set this Element's data.
   * \tparam E       The type of the element.
   */
  template <typename E>
  ripple_host_device constexpr Element(E&& element) noexcept
  : value{static_cast<T&&>(element)} {}

  T value; //!< The value of the element.
};

/*==--- [get implementation] -----------------------------------------------==*/

/**
 *  Helper function which can be decltype'd to get the type of the
 * element at the index I in a tuple.
 *
 * \note It should never actually be called.
 * \param  element The element to get the type of.
 * \tparam I       The index of the element in the tuple.
 * \tparam T       The type to extract from the element.
 */
template <size_t I, typename T>
ripple_host_device constexpr inline auto
type_extractor(Element<I, T> e) noexcept -> T {
  return T{};
}

/**
 * Gets a constant rvalue-reference to an Element.
 * \param  e The element to get the reference to.
 * \tparam I The index of the element in the tuple.
 * \tparam T The type to extract from the element.
 * \return A const rvalue reference to the element.
 */
template <size_t I, typename T>
ripple_host_device constexpr inline auto
get_impl(const Element<I, T>&& e) noexcept
  -> const std::remove_reference_t<T>&& {
  return e.value;
}

/**
 * Gets a constant lvalue-reference to an Element.
 * \param  e The element to get the reference to.
 * \tparam I The index of the element in the tuple.
 * \tparam T The type to extract from the element./
 * \return A const reference to the element.
 */
template <size_t I, typename T>
ripple_host_device constexpr inline auto
get_impl(const Element<I, T>& e) noexcept -> const std::remove_reference_t<T>& {
  return e.value;
}

/**
 * Gets an lvalue-reference to an Element.
 * \param  e The element to get the reference to.
 * \tparam I The index of the element in the tuple.
 * \tparam T The type to extract from the element.
 * \return A reference to the element.
 */
template <size_t I, typename T>
ripple_host_device constexpr inline auto
get_impl(Element<I, T>& e) noexcept -> std::remove_reference_t<T>& {
  return e.value;
}

/**
 * Gets an rvalue-reference to an Element.
 * \param  e The element to get the reference to.
 * \tparam I The index of the element in the tuple.
 * \tparam T The type to extract from the element./
 * \return An rvalue reference to the element.
 */
template <size_t I, typename T>
ripple_host_device constexpr inline auto
get_impl(Element<I, T>&& e) noexcept -> std::remove_reference_t<T>&& {
  using DataType = decltype(e.value);
  return ripple_move(e.value);
}

/*==--- [tuple storage] ----------------------------------------------------==*/

/**
 *  Implementation of the storage for a tuple.
 *
 * The elements which are stored are laid out in memory in the same order in
 * which they appear in the variadic pack:
 *
 * ~~~{.cpp}
 *  TupleStorage<Indices, float, int, double> ...
 *
 * // Laid out as follows:
 * float   // 4 bytes
 * int     // 4 bytes
 * double  // 8 bytes
 * ~~~
 *
 * \tparam Is The indices of the tuple elements.
 * \tparam Ts   The types of the tuple elements.
 */
template <typename Is, typename... Ts>
struct TupleStorage;

/**
 * Specializatiion of the tuple storage implementation.
 *
 * \tparam Is The indices for the locations of the elements.
 * \tparam Ts The types of the elements.
 */
template <size_t... Is, typename... Ts>
struct TupleStorage<std::index_sequence<Is...>, Ts...> : Element<Is, Ts>... {
  /** Defines the size (number of elements) of the tuple storage. */
  static constexpr size_t elements = sizeof...(Ts);

  /**
   * Default constructor.
   */
  constexpr TupleStorage() = default;

  /**
   *  Constructor to set the elements of the tuple storage.
   *
   * \note This overload is selected when the elements are forwarding reference
   *       types.
   *
   * \param  elements  The elements to use to set the tuple.
   * \tparam Types     The types of the elements.
   */
  template <typename... Types>
  ripple_host_device constexpr TupleStorage(Types&&... elements) noexcept
  : Element<Is, Ts>{ripple_forward(elements)}... {}

  /**
   * Constructor to set the elements of the tuple storage.
   *
   * \note This overload is selected when the elements are const lvalue
   *       references, and copies the elements.
   *
   * \param  elements The elements to use to set the tuple.
   * \tparam Types    The types of the elements.
   */
  template <typename... Types>
  ripple_host_device TupleStorage(const Types&... elements) noexcept
  : Element<Is, Ts>{static_cast<const Ts&>(elements)}... {}
};

/*==--- [basic tuple] ------------------------------------------------------==*/

/**
 * A basic tuple class, which is essentially just a cleaner interface
 * for TupleStorage.
 *
 * The types are laid out in memory in the same order in which they are present
 * in the varaidic parameter pack, \sa TupleStorage.
 *
 * \tparam Ts The types of the tuple elements.
 */
template <typename... Ts>
struct BasicTuple
: TupleStorage<std::make_index_sequence<sizeof...(Ts)>, Ts...> {
  // clang-format off
  /** Alias for the index sequence. */
  using IndexSeq = std::make_index_sequence<sizeof...(Ts)>;
  /** Alias for the base type of the Tuple. */
  using Base     = TupleStorage<IndexSeq, Ts...>;
  // clang-format on

  /** Defines the number of elements in the tuple. */
  static constexpr size_t elements = Base::elements;

  /**
   * Default constructor to create an uninitialized tuple.
   */
  constexpr BasicTuple() = default;

  /**
   * Creates a BasicTuple from a variadic list of elements.
   *
   * \note This overload is called if the the elements are forwarding reference
   *       types.
   *
   * \param  elements  The elements for the tuple.
   * \tparam Types     The types of the elements for the tuple.
   */
  template <typename... Types>
  ripple_host_device explicit constexpr BasicTuple(Types&&... elements) noexcept
  : Base{ripple_move(elements)...} {}

  /**
   * Creates a BasicTuple from a variadic list of elements.
   *
   * \note This overload is called if the the elements are constant lvalue
   *       reference types.
   *
   * \param  elements The elements for the BasicTuple.
   * \tparam Types    The types of the elements for the BasicTuple.
   */
  template <typename... Types>
  ripple_host_device constexpr explicit BasicTuple(
    const Types&... elements) noexcept
  : Base{static_cast<const Ts&>(elements)...} {}
};

} // namespace ripple::detail

#endif // RIPPLE_CONTAINER_DETAIL_BASIC_TUPLE__HPP

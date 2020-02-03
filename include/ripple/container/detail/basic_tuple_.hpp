//==--- ripple/container/detail/basic_tuple_.hpp ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  basic_tuple_.hpp
/// \brief This file implements the lowest level functionality for a tuple
///        class.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_DETAIL_BASIC_TUPLE__HPP
#define RIPPLE_CONTAINER_DETAIL_BASIC_TUPLE__HPP

#include <ripple/utility/portability.hpp>
#include <utility>

namespace ripple::detail {

//==--- [element] ----------------------------------------------------------==//

/// Defines a struct to hold an element at a specific index in a container.
/// \tparam Index The index of the element being held.
/// \tparam T     The type of the element.
template <size_t Index, typename T>
struct Element {
  /// Default constructor which just initializes the stored type.
  ripple_host_device constexpr Element() = default;

  /// Constructor to set the value of the element to that of the \p element.
  /// \param  element The element to use to set this Element's data.
  /// \tparam E       The type of the element.
  template <typename E>
  ripple_host_device constexpr Element(E&& element) noexcept
  : value(std::forward<T>(element)) {}

  T value;  //!< The value of the element.
};

//==--- [get implementation] -----------------------------------------------==//

/// Helper function which can be decltype'd to get the type of the
/// element at the index I in a tuple. It should never actually be called.
/// \param  element The element to get the type of.
/// \tparam I       The index of the element in the tuple.
/// \tparam T       The type to extract from the element.
template <size_t I, typename T>
ripple_host_device constexpr inline auto type_extractor(Element<I, T> e) -> T {
  return T{}; 
}

/// Gets a constant rvalue-reference to an Element.
/// \param  e The element to get the reference to.
/// \tparam I The index of the element in the tuple.
/// \tparam T The type to extract from the element./
template <size_t I, typename T>
ripple_host_device constexpr inline auto get_impl(
  const Element<I, T>&& e
) -> const T&& {
  return e.value;
}

/// Gets a constant lvalue-reference to an Element.
/// \param  e The element to get the reference to.
/// \tparam I The index of the element in the tuple.
/// \tparam T The type to extract from the element./
template <size_t I, typename T>
ripple_host_device constexpr inline auto get_impl(
  const Element<I, T>& e
) -> const T& {
  return e.value;
}

/// Gets an lvalue-reference to an Element.
/// \param  e The element to get the reference to.
/// \tparam I The index of the element in the tuple.
/// \tparam T The type to extract from the element./
template <size_t I, typename T>
ripple_host_device constexpr inline auto get_impl(
  Element<I, T>& e
) -> std::remove_reference_t<T>& {
  return e.value;
}

/// Gets an rvalue-reference to an Element.
/// \param  e The element to get the reference to.
/// \tparam I The index of the element in the tuple.
/// \tparam T The type to extract from the element./
template <size_t I, typename T>
ripple_host_device constexpr inline auto get_impl(
  Element<I, T>&& e
) -> std::remove_reference_t<T>&& {
  return std::move(e.value);
}

//==--- [tuple storage] ----------------------------------------------------==//

/// Defines the implementation of the storage for Tuple types. The elements
/// which are stored are laid out in memory in the order in which they are
/// present in the parameter pack. For example:
///
/// \code{.cpp}
///   TupleStorage<Indices, float, int, double> ...
///
///   // Laid out as follows:
///   float   // 4 bytes
///   int     // 4 bytes
///   double  // 8 bytes
/// \endcode
///
/// \tparam Is The indices of the tuple elements.
/// \tparam Ts   The types of the tuple elements.
template <typename Is, typename... Ts> struct TupleStorage;

/// Specialization for the implementation of the TupleStorage.
/// \tparam Is The indices for the locations of the elements.
/// \tparam Ts The types of the elements.
template <size_t... Is, typename... Ts>
struct TupleStorage<std::index_sequence<Is...>, Ts...> : Element<Is, Ts>... {
  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines the size (number of elements) of the tuple storage.
  static constexpr size_t elements = sizeof...(Ts);

  /// Default constructor.
  ripple_host_device constexpr TupleStorage() = default;

  /// Constructor which sets the elements of the tuple. This overload is 
  /// selected when the elements are forwarding reference types.
  /// \param  elements  The elements to use to set the tuple.
  /// \tparam Types     The types of the elements.  
  template <typename... Types>
  ripple_host_device constexpr TupleStorage(Types&&... elements)
  : Element<Is, Ts>(std::forward<Ts>(elements))... {}

  /// Constructor which sets the elements of the tuple. This overload is
  /// selected when the elements are const lvalue references, in which case they
  /// need to be copied.
  /// \param  elements The elements to use to set the tuple.
  /// \tparam Types    The types of the elements.  
  template <typename... Types>
  ripple_host_device TupleStorage(const Types&... elements)
  : Element<Is, Ts>(std::forward<Ts>(elements))... {}
};

//==--- [basic tuple] ------------------------------------------------------==//

/// Defines a basic tuple class, which is essentially just a cleaner interface 
/// for TupleStorage. The types are laid out in memory in the same order in
/// which they appear in the parameter pack. For example:
///
/// ~~~cpp
///   BasicTuple<double, int, float> tuple(2.7, 4, 3.14f);
///
///   // Laid out as follows:
///   double  : 2.7    // 8 bytes
///   int     : 4      // 4 bytes
///   float   : 3.14   // 4 bytes
/// ~~~
///
/// \tparam Ts The types of the tuple elements.
template <typename... Ts>
struct BasicTuple : 
  TupleStorage<std::make_index_sequence<sizeof...(Ts)>, Ts...> {
  //==--- [alises] ---------------------------------------------------------==//

  /// Alias for the index sequence.
  using index_seq_t = std::make_index_sequence<sizeof...(Ts)>;
  /// Alias for the base type of the Tuple.
  using base_t      = TupleStorage<index_seq_t, Ts...>;

  //==--- [constants] -------------------------------------------------------==//
  
  /// Defines the size (number of elements) of the BasicTuple.
  static constexpr size_t elements = base_t::elements;

  //==--- [construction] ---------------------------------------------------==//
  
  /// Default constructor to create an uninitialized tuple.
  ripple_host_device constexpr BasicTuple() = default;

  /// Creates a BasicTuple from a variadic list of elements. This overload is
  /// called if the the \p elements are forwarding reference types.
  /// \param  elements  The elements for the tuple.
  /// \tparam Types     The types of the elements for the tuple.
  template <typename... Types>
  ripple_host_device explicit constexpr BasicTuple(Types&&... elements)
  : base_t(std::forward<Ts>(elements)...) {}

  /// Creates a BasicTuple from a variadic list of elements. This overload is
  /// called if the the \p elements are constant lvalue reference types.
  /// \param[in] elements The elements for the BasicTuple.
  /// \tparam    Types    The types of the elements for the BasicTuple.  
  template <typename... Types>
  ripple_host_device constexpr explicit BasicTuple(const Types&... elements)
  : base_t(std::forward<Ts>(elements)...) {}
};  

} // namespace ripple::detail

#endif // RIPPLE_CONTAINER_DETAIL_BASIC_TUPLE__HPP

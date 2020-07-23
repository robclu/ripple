//==--- ripple/io/printable_traits.hpp --------------------- -*- c++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  printable_traits.hpp
/// \brief This file defines traits for printable types.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_IO_PRINTABLE_TRAITS_HPP
#define RIPPLE_IO_PRINTABLE_TRAITS_HPP

#include <ripple/core/utility/portability.hpp>

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// The Printable interface defines a static interface for types which can
/// print data using a specific interface.
/// \tparam Impl the implementation type for the interface.
template <typename Impl>
class Printable;

/// The PrintableElement type holdes a name and a values of an element which can
/// then be printed.
class PrintableElement;

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T is Printable, otherwise returns false.
/// \tparam T The type to determine if is Printable.
template <typename T>
static constexpr auto is_printable_v =
  std::is_base_of_v<Printable<std::decay_t<T>>, std::decay_t<T>>;

/// Returns true if the type T is a PrintableElement, otherwise returns false.
/// \tparam T The type to determine if is Printable.
template <typename T>
static constexpr auto is_printable_element_v =
  std::is_same_v<PrintableElement, std::decay_t<T>>;

} // namespace ripple

#endif // RIPPLE_IO_PRINTABLE_TRAITS_HPP
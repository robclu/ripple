//==--- ripple/io/printable_element.hpp -------------------- -*- c++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  printable_element.hpp
/// \brief This file defines a class to represent a printable element.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_IO_PRINTABLE_ELEMENT_HPP
#define RIPPLE_IO_PRINTABLE_ELEMENT_HPP

#include "printable_traits.hpp"
#include "../utility/forward.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace ripple {

/**
 * The PrintableElement type holdes a name and values for an element which can
 * be printed.
 */
class PrintableElement {
 public:
  // clang-format off
  /** Defines the type of the values for the element. */
  using Value          = double;
  /** Defines the type of the container used to store the element values. */
  using ValueContainer = std::vector<Value>;
  // clang-format on

  /** Defines the possible kinds of attributes. */
  enum AttributeKind : uint8_t {
    scalar, //!< Scalar data.
    vector, //!< Vector data.
    invalid //!< Invalid data.
  };

 private:
  ValueContainer values_ = {}; //!< Element values.

 public:
  std::string   name = "";                     //!< Element name.
  AttributeKind kind = AttributeKind::invalid; //!< Element kind.

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor which creates an empty element.
   */
  PrintableElement() noexcept = default;

  /**
   * Creates a printable element with the given name, kind, and values.
   * \param  name   The name of the element.
   * \param  kind   The kind of the element.
   * \param  values Values for the element.
   * \tparam Values The types of the values.
   */
  template <typename... Values>
  PrintableElement(
    std::string name, AttributeKind kind, Values&&... values) noexcept
  : values_{ripple_forward(values)...}, name{ripple_move(name)}, kind{kind} {}

  //==--- [interface] ------------------------------------------------------==//

  /**
   * Compares this element against another element.
   * \param other The other element to compare against.
   * \return true if the elements are equal.
   */
  auto operator==(const PrintableElement& other) const noexcept -> bool;

  /**
   * Compares this element against another element.
   * \param other The other element to compare against.
   * \return true if the eleements are not the same.
   */
  auto operator!=(const PrintableElement& other) const noexcept -> bool;

  /**
   * Adds a value to the printable element.
   * \param value The value to add to the element.
   */
  auto add_value(const Value& value) noexcept -> void;

  /**
   * Gets a reference to the values to print. If the kind is a vector, then
   * the vector requires 3 elements, which will be default added as zero
   * components if there are not enough components.
   * \return A container of values.
   */
  auto values() const noexcept -> const ValueContainer&;

  /**
   * Gets the first value from the values, or the only value if there is
   * only one.
   * \return The first value.
   */
  auto first_value() const noexcept -> Value;

  /**
   * Determines if the attribute is invalid.
   * \return true if the element is invalid.
   */
  auto is_invalid() const noexcept -> bool;

  /**
   * Gets a PrintableElement with the name 'not found' and an invalid kind.
   * \return An invalid printable element.
   */
  static auto not_found() -> PrintableElement;
};

/**
 * Gets a printable element for the type, with the given name. If the type is
 * a printable element, is just returns that, otherwise it creates a default
 * printable element.
 * \param  dat  The data to get the printable element from.
 * \param  name The name of the element to get.
 * \param  args Additional arguments to get the element.
 * \tparam T    The type of the data.
 * \tparam Args The types of the arguments.
 * \return A printable element.
 */
template <typename T, typename... Args>
auto printable_element(T&& data, const char* name, Args&&... args) noexcept
  -> PrintableElement {
  if constexpr (is_printable_v<T>) {
    return data.printable_element(name, ripple_forward(args)...);
  } else {
    return PrintableElement{
      name, PrintableElement::AttributeKind::scalar, data};
  }
}

} // namespace ripple

#endif // RIPPLE_IO_PRINTABLE_ELEMENT_HPP
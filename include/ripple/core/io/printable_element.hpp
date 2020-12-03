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
#include <cstring>
#include <string>
#include <vector>

namespace ripple {

/// The PrintableElement type holdes a name and values for an element which can
/// then be printed.
class PrintableElement {
 public:
  //==--- [aliases] --------------------------------------------------------==//

  // clang-format off
  /// Defines the type of the values for the element.
  using value_t           = double;
  /// Defines the type of the container used to store the element values.
  using value_container_t = std::vector<value_t>;
  // clang-format on

  /// Defines the possible kinds of attributes.
  enum AttributeKind : uint8_t {
    scalar, //!< Scalar data.
    vector, //!< Vector data.
    invalid //!< Invalid data.
  };

 private:
  value_container_t _values = {}; //!< Element values.

 public:
  std::string   name = "";                     //!< Element name.
  AttributeKind kind = AttributeKind::invalid; //!< Element kind.

  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor which creates an empty element.
  PrintableElement() = default;

  /// Creates a printable element with the \p name and \p kind, with \p values
  /// values.
  /// \param  name   The name of the element.
  /// \param  kind   The kind of the element.
  /// \param  values Values for the element.
  /// \tparam Values The types of the values.
  template <typename... Values>
  PrintableElement(
    std::string name, AttributeKind kind, Values&&... values) noexcept
  : _values{static_cast<value_t>(values)...},
    name{std::move(name)},
    kind{kind} {}

  //==--- [comparison] -----------------------------------------------------==//

  /// Compares this element against another element, returning true if the
  /// element names match, otherwise returning false.
  auto operator==(const PrintableElement& other) const noexcept -> bool {
    return std::memcmp(&other.name[0], &name[0], name.length()) == 0;
  }

  /// Compares this element against another element, returning true if the
  /// element names don't match, otherwise returning true.
  auto operator!=(const PrintableElement& other) const noexcept -> bool {
    return !(*this == other);
  }

  //==--- [interface] ------------------------------------------------------==//

  /// Adds a value to the printable element.
  /// \param value The value to add to the element.
  auto add_value(const value_t& value) noexcept -> void {
    _values.emplace_back(value);
  }

  /// Returns a reference to the values to print. If the kind is a vector, then
  /// the vector rquires 3 elements, which will be default added as zero
  /// components if there are not enough components.
  auto values() const noexcept -> const value_container_t& {
    return _values;
  }

  /// Returns the first value from the values, or the only value if there is
  /// only one.
  auto first_value() const noexcept -> value_t {
    return _values[0];
  }

  /// Returns true if the element is in valid.
  auto is_invalid() const noexcept -> bool {
    return kind == AttributeKind::invalid;
  }

  //==--- [invalid interface] ----------------------------------------------==//

  /// Returns a PrintableElement with the name 'not found' and an invalid kind.
  static auto not_found() -> PrintableElement {
    return PrintableElement("not found", AttributeKind::invalid);
  }
};

/**
 * Returns a printable element for the type, with the given name. If the type is
 * a printable element, is just returns that, otherwise it creates a default
 * printable element.
 */
template <typename T, typename... Args>
decltype(auto)
printable_element(T&& data, const char* name, Args&&... args) noexcept {
  if constexpr (is_printable_v<T>) {
    return data.printable_element(name, ripple_forward(args)...);
  } else {
    return PrintableElement{
      name, PrintableElement::AttributeKind::scalar, data};
  }
}

} // namespace ripple

#endif // RIPPLE_IO_PRINTABLE_ELEMENT_HPP
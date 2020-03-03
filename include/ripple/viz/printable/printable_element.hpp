//==--- ripple/viz/printable/printable.hpp ----------------- -*- c++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  printable.hpp
/// \brief This file defines an interface for a type which is printable.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_VIZ_PRINTABLE_PRINTABLE_ELEMENT_HPP
#define RIPPLE_VIZ_PRINTABLE_PRINTABLE_ELEMENT_HPP

#include "printable_traits.hpp"
#include <string>
#include <vector>

namespace ripple::viz {

/// The PrintableElement type holdes a name and a values of an element which can
/// then be printed.
class PrintableElement {
 public:
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of the values for the element.
  using value_t           = double;
  /// Defines the type of the container used to store the element values.
  using value_container_t = std::vector<value_t>;

  /// Defines the possible kinds of attributes.
  enum AttributeKind : uint8_t {
    scalar, //!< Scalar data.
    vector  //!< Vector data.
  };

  //==--- [construction] ---------------------------------------------------==//
  
  /// Creates a printable element with the \p name and \p kind, with \p values
  /// values.
  /// \param  name   The name of the element.
  /// \param  kind   The kind of the element.
  /// \param  values Values for the element.
  /// \tparam Values The types of the values.
  template <typename... Values>
  PrintableElement(std::string name, AttributeKind kind, Values&&... values)
  : _values{static_cast<value_t>(values)...},
    _name{std::move(name)}, 
    _kind{kind} {}

  //==--- [interface] ------------------------------------------------------==//
  
  /// Adds a value to the printable element.
  /// \param value The value to add to the element.
  auto add_value(const value_t& value) -> void {
    _values.emplace_back(value);
  }

  /// Returns a reference to the values to print. If the kind is a vector, then
  /// the vector rquires 3 elements, which will be default added as zero
  /// components if there are not enough components.
  auto values() ->  value_container_t& {
    if (_kind == AttributeKind::vector && _values.size() < 3) {
      while (_values.size() < 3) {
        _values.emplace_back(value_t{0});
      }
    }
    return _values;
  }

  /// Returns a reference to the kind of the element.
  auto kind() -> AttributeKind& {
    return _kind;
  }

  /// Returns the kind of the printable element.
  auto kind() const -> AttributeKind {
    return _kind;
  }

  /// Returns a reference to the name of the element.
  auto name() -> std::string& {
    return _name;
  }

  /// Returns a constant reference to the name of the element.
  auto name() const -> const std::string& {
    return _name;
  }

 private:
  value_container_t _values;                       //!< Values for the element.
  std::string       _name;                         //!< Name of the element.
  AttributeKind     _kind = AttributeKind::scalar; //!< The kind of the element.
};

} // namespace ripple::viz

#endif // RIPPLE_VIZ_PRINTABLE_PRINTABLE_ELEMENT_HPP

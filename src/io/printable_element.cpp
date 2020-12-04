//==--- ripple/src/io/printable_element.cpp ---------------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  printable_element.cpp
/// \brief This file implements the printable element class.
//
//==------------------------------------------------------------------------==//

#include "../include/ripple/core/io/printable_element.hpp"

namespace ripple {

auto PrintableElement::operator==(const PrintableElement& other) const noexcept
  -> bool {
  return std::memcmp(&other.name[0], &name[0], name.length()) == 0;
}

auto PrintableElement::operator!=(const PrintableElement& other) const noexcept
  -> bool {
  return !(*this == other);
}

auto PrintableElement::add_value(const Value& value) noexcept -> void {
  values_.emplace_back(value);
}

auto PrintableElement::values() const noexcept -> const ValueContainer& {
  return values_;
}

auto PrintableElement::first_value() const noexcept -> Value {
  return values_[0];
}

auto PrintableElement::is_invalid() const noexcept -> bool {
  return kind == AttributeKind::invalid;
}

auto PrintableElement::not_found() -> PrintableElement {
  return PrintableElement("not found", AttributeKind::invalid);
}

} // namespace ripple
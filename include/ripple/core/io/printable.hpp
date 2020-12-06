//==--- ripple/io/printable.hpp ---------------------------- -*- C++ -*- ---==//
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
/// \brief this file defines an interface for a type which is printable.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_IO_PRINTABLE_PRINTABLE_HPP
#define RIPPLE_IO_PRINTABLE_PRINTABLE_HPP

#include "printable_element.hpp"
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/**
 * The Printable interface defines a static interface for types which can
 * print data using a specific interface.
 * \tparam Impl the implementation type for the interface.
 */
template <typename Impl>
class Printable {
  /**
   * Gets a const pointer to the implementation.
   * \return A const pointer to the implementation.
   */
  ripple_host_device constexpr auto impl() const -> const Impl* {
    return static_cast<const Impl*>(this);
  }

 public:
  /**
   * Gets a printable element with the given name and using the optional
   * arguments if necessary.
   * \param  name The name of the element to get.
   * \param  args Optional arguments used to get the elements.
   * \tparam Args The type of the arguments.
   * \return A printable element.
   */
  template <typename... Args>
  auto printable_element(const char* name, Args&&... args) const noexcept
    -> PrintableElement {
    return impl()->printable_element_impl(name, std::forward<Args>(args)...);
  }
};

} // namespace ripple

#endif // RIPPLE_IO_PRINTABLE_PRINTABLE_HPP
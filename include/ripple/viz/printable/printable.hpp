//==--- ripple/viz/printable/printable.hpp ----------------- -*- C++ -*- ---==//
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

#ifndef RIPPLE_VIZ_PRINTABLE_PRINTABLE_HPP
#define RIPPLE_VIZ_PRINTABLE_PRINTABLE_HPP

#include "printable_element.hpp"
#include <ripple/core/utility/portability.hpp>

namespace ripple::viz {

/// The Printable interface defines a static interface for types which can 
/// print data using a specific interface.
/// \tparam Impl the implementation type for the interface.
template <typename Impl>
class Printable {
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
  /// Returns true if the type has an element which can be printed and has the
  /// name \p name.
  /// \param name The name of the element to determine if is present in the type
  ///             implementing the printable interface.
  ripple_host_only auto has_printable_element(const char* name) const
  -> bool {
    return impl()->has_printable_element(name);
  }

  /// Returns a printable element with the name \p name, using the arguments \p
  /// args if necessary.
  /// \param  name The name of the element to get.
  /// \param  args Optional arguments used to get the elements.
  /// \tparam Args The type of the arguments.
  template <typename... Args>
  ripple_host_only auto printable_element(const char* name, Args&&... args)
  const -> PrintableElement {
    return impl()->printable_element(name, std::forward<Args>(args)...);
  }
};

} // namespace ripple::viz

#endif // RIPPLE_VIZ_PRINTABLE_PRINTABLE_HPP

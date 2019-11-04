//==--- ripple/storage/stridable_layout.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  stridable_layout.hpp
/// \brief This file defines a static inteface for classes which can have a
///        stridable layout.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_AUTO_LAYOUT_HPP
#define RIPPLE_STORAGE_AUTO_LAYOUT_HPP

#include "storage_layout.hpp"
#include "storage_traits.hpp"

namespace ripple {

/// The AutoLayout class defines a static interface for classes which define the
/// classes to layout, but which allow the allocation and layout of bulk data
/// for the type to be automatically performed by the container. This is useful
/// for types on which computationally intense processing will be performed so
/// that the storage can be allocated and laid out optimally for the processing
/// platform.
/// \tparam Impl The implementation of the interface.
template <typename Impl> 
struct AutoLayout {
 private:
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a constant pointer to the implementation type.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation type.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
  /// Returns the desired kind of layout.
  ripple_host_device constexpr auto layout() const -> LayoutKind {
    return impl()->layout();
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_AUTO_LAYOUT_HPP

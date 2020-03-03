//==--- ripple/viz/io/writer.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  writer.hpp
/// \brief This file defines an implementation for a static interface for types
///        which can write data in different formats.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_VIZ_IO_WRITER_HPP
#define RIPPLE_VIZ_IO_WRITER_HPP

#include <ripple/core/utility/portability.hpp> 

namespace ripple::viz {

/// The Writer interface defines a static interface for types which can write
/// data in a certain format.
/// \tparam Impl The implementation type for the interface.
template <typename Impl>
class Writer {
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
};

} // namespace ripple::viz

#endif // RIPPLE_VIZ_IO_WRITER_HPP


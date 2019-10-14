//==--- streamline/container/array.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Streamline.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array.hpp
/// \brief This file defines an interface for arrays.
//
//==------------------------------------------------------------------------==//

#ifndef STREAMLINE_CONTAINER_ARRAY_HPP
#define STREAMLINE_CONTAINER_ARRAY_HPP

#include "array_traits.hpp"
#include <streamline/utility/portability.hpp>

namespace streamline {

/// The Array class defines an interface to which all specialized
/// implementations must conform. The implementation is provided by the template
/// type Impl.
/// \tparam Impl The implementation of the array interface.
template <typename Impl>
struct Array {
 private:
  // Defines the type of the implementation.
  using impl_t = Impl;

 public:
  /// Returns the value at position \p i in the array.
  /// \param[in]
  streamline_host_device constexpr auto operator[](std::size_t i) 
  -> typename ArrayTraits<Impl>::value_t& {
    return impl()->operator[](i);  
  }

  /// Returns the value at position \p i in the array.
  streamline_host_device constexpr auto operator[](std::size_t i) const 
  -> const typename ArrayTraits<Impl>::value_t& {
    return impl()->operator[](i);  
  }
    
  /// Returns the number of elements in the array.
  streamline_host_device constexpr auto size() const -> std::size_t {
    return ArrayTraits<Impl>::size;
  }
    
 private:
  /// Returns a pointer to the implementation of the interface.
  streamline_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }
    
  /// Returns a pointer to constant implementation of the interface.
  streamline_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }
};

} // namespace streamline

#endif // STREAMLINE_CONTAINER_ARRAY_HPP

//==--- ripple/storage/storage_element_traits.hpp ---------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_element_traits.hpp
/// \brief This file traits for elements to be stored.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STORAGE_ELEMENT_TRAITS_HPP
#define RIPPLE_STORAGE_STORAGE_ELEMENT_TRAITS_HPP

#include <ripple/utility/portability.hpp>
namespace ripple {

/// The StorageElement class defines a class which can specify a type to store,
/// and the number of those type to store. Its intended use is to define a
/// description of the data for a class, which can then be used by multiple
/// storage mechanisms to store the data with differnt layouts.
/// \tparam T      The type of the data for the element.
/// \tparam Values The number of value of type T to store.
template <typename T, std::size_t Values>
class StorageElement {
 public:
  /// Returns the number of elements to store.
  ripple_host_device constexpr auto size() -> std::size_t {
    return Values;
  }
};

/// The StorageElementTraits struct defines traits for elements which can be
/// stored. This is the default case which is used to define the storage traits
/// for any type that does not have specialized storage.
/// \tparam T The type to specify the storage element traits for.
template <typename T>
struct StorageElementTraits {
  /// Defines the value type of the element, which is the type to store.
  using value_t = T;
  /// Defines the number of values of the element to store.
  static constexpr auto num_elements = 1;
  /// Defines the byte size required to allocate the element.
  static constexpr auto byte_size    = sizeof(value_t);
};

/// Specialization for a StorageElement.
template <typename T, std::size_t Values> 
struct StorageElementTraits<StorageElement<T, Values>> {
  /// Defines the value type of the element, which is the type to store.
  using value_t = T;
  /// Defines the number of values of the element to store.
  static constexpr auto num_elements = Values;
  /// Defines the byte size required to allocate the element.
  static constexpr auto byte_size    = sizeof(value_t) * num_elements;
};
  

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_ELEMENT_TRAITS_HPP

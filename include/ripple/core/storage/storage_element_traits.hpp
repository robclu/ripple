//==--- ripple/core/storage/storage_element_traits.hpp ---------- -*- C++ -*- ---==//
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

#include "storage_layout.hpp"
#include <ripple/core/utility/portability.hpp>

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
  /// Defines the alignment size required for the type.
  static constexpr auto align_size   = alignof(value_t);
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
  /// Defines the alignment requirement for the storage.
  static constexpr auto align_size   = alignof(value_t);
};

//==--- [is storage element] -----------------------------------------------==//

namespace detail {

/// Defines a class to determine of the type T is a layout type.
/// \tparam T The type to determine if is a storage layout type.
template <typename T>
struct IsStorageLayout : std::false_type {
  /// Defines that the type is not a storage layout type.
  static constexpr auto value = false;
};

/// Specialization for the case of the IsStorageLayout struct for the case the
/// type to check is a StorageLayout.
/// \tparam Layout The kind of the layout for the storage.
template <LayoutKind Layout>
struct IsStorageLayout<StorageLayout<Layout>> : std::true_type {
  /// Defines that the type is a storage layout type.
  static constexpr auto value = true;
};

/// Defines a class to determine of the type T is a storage element type.
/// \tparam T The type to determine if is a storage element type.
template <typename T>
struct IsStorageElement : std::false_type {
  /// Defines that the type is not a storage element type.
  static constexpr auto value = false;
};

/// Specialization for the case of the IsStorageElement struct for the case the
/// type to check is a StorageElement.
/// \tparam T     The type for the element.
/// \tparam Value The number of values for the type.
template <typename T, std::size_t Values>
struct IsStorageElement<StorageElement<T, Values>> : std::true_type {
  /// Defines that the type is a storage element type.
  static constexpr auto value = true;
};


/// Defines a struct to determine if the type T has a storage layout type
/// template parameter.
/// \tparam T The type to determine if has a storage layout parameter.
template <typename T>
struct HasStorageLayout {
  /// Returns that the type does not have a storage layout paramter.
  static constexpr auto value = false;
};

/// Specialization for the case that the type has template parameters.
/// \tparam T  The type to determine if has a storage layout parameter.
/// \tparam Ts The types for the type T.
template <template <class...> typename T, typename... Ts>
struct HasStorageLayout<T<Ts...>> {
  /// Returns that the type does not have a storage layout paramter.
  static constexpr auto value = std::disjunction_v<IsStorageLayout<Ts>...>;
};

} // namespace detail

/// Returns true if the type T is a StorageElement type, otherwise returns
/// false.
/// \tparam T The type to determine if is a StoageElement type.
template <typename T>
static constexpr auto is_storage_element_v = 
  detail::IsStorageElement<std::decay_t<T>>::value;

//==-- [aliases] -----------------------------------------------------------==//

/// Alias for storage element traits.
/// \tparam T The type to get the traits for.
template <typename T>
using storage_element_traits_t = StorageElementTraits<std::decay_t<T>>;

//==--- [enables] ----------------------------------------------------------==//

/// Define a valid type if the type T is a StorageElement, otherwise does not
/// define a valid type.
/// \tparam T The type to base the enable on.
template <typename T>
using storage_element_enable_t = std::enable_if_t<is_storage_element_v<T>, int>;

/// Define a valid type if the type T is not a StorageElement, otherwise does
/// not define a valid type.
/// \tparam T The type to base the enable on.
template <typename T>
using non_storage_element_enable_t =
  std::enable_if_t<!is_storage_element_v<T>, int>;
  
} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_ELEMENT_TRAITS_HPP

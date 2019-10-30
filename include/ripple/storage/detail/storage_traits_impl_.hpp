//==--- ripple/storage/detail/storage_traits_impl_.hpp ----- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_traits_impl_.hpp
/// \brief This file defines implementation details for storage traits.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_DETAIL_STORAGE_TRAITS_IMPL_HPP
#define RIPPLE_STORAGE_DETAIL_STORAGE_TRAITS_IMPL_HPP

#include "../storage_element_traits.hpp"
#include "../storage_layout.hpp"
#include <ripple/utility/type_traits.hpp>

namespace ripple {
namespace detail {

//==--- [is storage layout] ------------------------------------------------==//

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

//==--- [is storage element] -----------------------------------------------==//

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

//==--- [has storage layout] -----------------------------------------------==//

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


}} // namespace ripple::detail

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

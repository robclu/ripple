//==--- ripple/storage/storage_traits.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_traits.hpp
/// \brief This file defines traits for storage.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STORAGE_TRAITS_HPP
#define RIPPLE_STORAGE_STORAGE_TRAITS_HPP

#include "detail/storage_traits_impl_.hpp"
#include "storage_element_traits.hpp"

namespace ripple {

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T is a StorageLayout type, otherwise returns false.
/// \tparam T The type to determine if is a StoageLayout type.
template <typename T>
static constexpr auto is_storage_layout_v = 
  detail::IsStorageLayout<std::decay_t<T>>::value;

/// Returns true if the type T is a StorageElement type, otherwise returns
/// false.
/// \tparam T The type to determine if is a StoageElement type.
template <typename T>
static constexpr auto is_storage_element_v = 
  detail::IsStorageElement<std::decay_t<T>>::value;

/// Returns true if the type T has template parameters and one of the template
/// parameters is a StorageLayout type.
///
/// \note This will not work if the class has template parameters which are not
///       types. If this is required, the numeric template parameters can be
///       wrapped in the Num or Int64 classes.
///
/// \tparam T The type to determine if has a storage layout parameter.
template <typename T>
static constexpr auto has_storage_layout_v =
  detail::HasStorageLayout<std::decay_t<T>>::value;    

//==-- [aliases] -----------------------------------------------------------==//

/// Defines an alias for storage which is laid out contiguously.
using contiguous_layout_t = StorageLayout<LayoutKind::contiguous>;
/// Defines an alias for storage which is laid out strided.
using strided_layout_t    = StorageLayout<LayoutKind::strided>;

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

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

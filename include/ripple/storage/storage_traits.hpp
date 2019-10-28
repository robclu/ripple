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

namespace ripple {

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T is a StorageLayout type, otherwise returns false.
/// \tparam T The type to determine if is a StoageLayout type.
template <typename T>
static constexpr auto is_storage_layout_v = 
  detail::IsStorageLayout<std::decay_t<T>>::value;

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

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

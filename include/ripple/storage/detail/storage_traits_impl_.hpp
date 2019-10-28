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

#include "../storage_layout.hpp"

namespace ripple {
namespace detail {

/// Defines a class to determine of the type T is a layout type.
/// \tparam T The type to determine if is a storage layout type.
template <typename T>
struct IsStorageLayout {
  /// Defines that the type is not a storage layout type.
  static constexpr auto value = false;
};

/// Specialization for the case of the IsStorageLayout struct for the case the
/// type to check is a StorageLayout.
/// \tparam Layout The kind of the layout for the storage.
template <LayoutKind Layout>
struct IsStorageLayout<StorageLayout<Layout>> {
  /// Defines that the type is a storage layout type.
  static constexpr auto value = true;
};

}} // namespace ripple::detail

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

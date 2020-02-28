//==--- ripple/core/storage/storage_descriptor.hpp -------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_descriptor.hpp
/// \brief This file defines a type to describe storage for a class.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STORAGE_DESCRIPTOR_HPP
#define RIPPLE_STORAGE_STORAGE_DESCRIPTOR_HPP

#include "contiguous_storage_view.hpp"
#include "owned_storage.hpp"
#include "storage_layout.hpp"
#include "strided_storage_view.hpp"
#include "storage_layout.hpp"
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/// The StorageDescriptor class can be used to define the types to store. This
/// is an empty struct, and is just used to wrap the layout kind and types so
/// that it can bbe specialized for specific traits.
/// \tparam Layout The layout for the data.
/// \tparam Ts     The types to store.
template <typename Layout, typename... Ts>
struct StorageDescriptor {
  /// Defines the type of the layout for the descriptor.
  static constexpr auto layout = Layout::value;

  /// Defines the type of contigous view storage.
  using contig_view_storage_t  = ContiguousStorageView<Ts...>;
  /// Defines the type of strided view storage.
  using strided_view_storage_t = StridedStorageView<Ts...>;
  /// Defines the type of owned storage.
  using owned_storage_t        = OwnedStorage<Ts...>;

  /// Defines which type of storage to use, based on the type of the layout.
  using storage_t = std::conditional_t<
    layout == LayoutKind::contiguous_view,
    contig_view_storage_t, 
    std::conditional_t<
      layout == LayoutKind::strided_view,
      strided_view_storage_t,
      owned_storage_t
    >
  >;
};

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_DESCRIPTOR_HPP

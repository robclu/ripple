//==--- ripple/storage/storage_layout.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_layout.hpp
/// \brief This file defines a class to represent storage layouts.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STORAGE_LAYOUT_HPP
#define RIPPLE_STORAGE_STORAGE_LAYOUT_HPP

namespace ripple {

/// The LayoutKind enum defines the possible types of layouts.
enum class LayoutKind : uint8_t {
	contiguous = 0, //!< Specifier for data which must be laid out contiguously.
  strided    = 1  //!< Specifier for data which must be laid out strided.
};

/// The StorageLayout struct defines a type which represents the the layout for
/// storage.
/// \tparam Layout The kind of the layout.
template <LayoutKind Layout>
struct StorageLayout { 
  /// Defines the value of the layout for the storage.
  static constexpr auto value = Layout;
};

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_LAYOUT_HPP

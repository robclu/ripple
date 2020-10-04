//==--- ripple/core/storage/storage_layout.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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

#include <ripple/core/utility/portability.hpp>

namespace ripple {

/** The LayoutKind enum defines the possible types of layouts. */
enum class LayoutKind : uint8_t {
  contiguous_view  = 0, //!< Contiguous, viewed (not owned) layout.
  strided_view     = 1, //!< Strided, viewed (not owned) layout.
  contiguous_owned = 2, //!< Contiguous, owned layout.
  none             = 3  //!< No storage layout specified.
};

/**
 * The StorageLayout struct defines a type which represents the the layout for
 * storage, as a type.
 * \tparam Layout The kind of the layout.
 */
template <LayoutKind Layout>
struct StorageLayout {
  /** Defines the value of the layout for the storage. */
  static constexpr auto value = Layout;
};

// clang-format off
/** Alias for a type which defines contigous viewed storage. */
using ContiguousView  = StorageLayout<LayoutKind::contiguous_view>;
/** Alias for a type which defines strided viewed storage. */
using StridedView     = StorageLayout<LayoutKind::strided_view>;
/** Alias for a type which defines contiguous owned storage. */
using ContiguousOwned = StorageLayout<LayoutKind::contiguous_owned>;
/** Alias for a type which defines no storage layout. */
using NoLayout        = StorageLayout<LayoutKind::none>;
// clang-format on

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_LAYOUT_HPP

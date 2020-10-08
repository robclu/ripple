//==--- ripple/core/storage/polymorphic_layout.hpp --------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  polymorphic_layout.hpp
/// \brief This file defines a class which can be inherited to make the layout
///        of its data polymorphic.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_POLYMORPHIC_LAYOUT_HPP
#define RIPPLE_STORAGE_POLYMORPHIC_LAYOUT_HPP

namespace ripple {

/**
 * The PolymorphicLayout class defines a class which can be used as an empty
 * base class to define that the layout of the data for the class is
 * polymorphic.
 *
 * \tparam Impl The implementation type with a polymorphic layout.
 */
template <typename Impl>
struct PolymorphicLayout {};

} // namespace ripple

#endif // RIPPLE_STORAGE_POLYMORPHIC_LAYOUT_HPP

//==--- ripple/core/storage/stridable_layout.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  stridable_layout.hpp
/// \brief This file defines a static inteface for classes which can have a
///        stridable layout.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STRIDABLE_LAYOUT_HPP
#define RIPPLE_STORAGE_STRIDABLE_LAYOUT_HPP

namespace ripple {

/// The StridableLayout class defines a static interface for classes to implement
/// for which is might be beneficial to allocate the class data in a strided
/// layout -- essentially any class which might be used for processing on either
/// the GPU or using AVX -- which is more performant.
///
/// Inheriting this static interface will allow any containers which can use the
/// strided allocators to do so where appropriate.
///
/// \tparam Impl The implementation of the interface.
template <typename Impl> 
struct StridableLayout {};

} // namespace ripple

#endif // RIPPLE_STORAGE_STRIDABLE_LAYOUT_HPP

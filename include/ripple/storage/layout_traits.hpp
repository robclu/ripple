//==--- ripple/storage/layout_traits.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  layout_traits.hpp
/// \brief This file defines traits for the layout of a type.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_LAYOUT_TRAITS_HPP
#define RIPPLE_STORAGE_LAYOUT_TRAITS_HPP

#include "detail/storage_traits_impl_.hpp"
#include "default_storage.hpp"
#include "pointer_wrapper.hpp"
#include "storage_layout.hpp"

namespace ripple {

/// Defines layout traits for the the T. This is the default implementation for
/// the case that the type T does not implement the StridableLayout interface.
/// \tparam T                 The type to get the layout traits for.
/// \tparam IsStridableLayout If the type is an StridableLayout type.
template <typename T, bool IsStridableLayable>
struct LayoutTraits {
  //==--- [constants] ------------------------------------------------------==//

  /// Defines if the type T is a StridableLayout type.
  static constexpr bool is_stridable_layout = false;
  /// Defines the type of the layout for T.
  static constexpr auto layout_kind         = LayoutKind::none;
    /// True if the Layout is a LayoutKind::strided_view.
  static constexpr auto is_strided_view     = false;

  //==--- [traits] ---------------------------------------------------------==//

  /// Defines the value type of T.
  using value_t          = std::decay_t<T>;
  /// Defines the type of the allocator for type T.
  using allocator_t      = typename DefaultStorage<T>::allocator_t;
  /// Defines the type for making a copy of T.
  using iter_copy_t      = value_t;
  /// Defines the type to use when storing in an iterator.
  using iter_storage_t   = value_t*;
  /// Defines the type when referencing from an iterator.
  using iter_ref_t       = value_t&;
  /// Defines the type of a const reference for an iterator.
  using iter_const_ref_t = const value_t&;
  /// Defines the type of a pointer for an iterator.
  using iter_ptr_t       = value_t*;
  /// Defines thet type of a const pointer for an iterator.
  using iter_const_ptr_t = const value_t*;
};

/// Defines layout traits for the the T when the type implements the
/// StridableLayout interface.
/// \tparam T The type to get the layout traits for.
template <typename T>
struct LayoutTraits<T, true> {
 private:
  /// Defines the type of the descriptor for the type T.
  using descriptor_t      = typename T::descriptor_t;
  /// Defines the type for strided view storage.
  using strided_view_t    = typename descriptor_t::strided_view_storage_t;
  /// Defines the type for contiguous view storage.
  using contig_view_t     = typename descriptor_t::contig_view_storage_t;
  /// Defines the type T with contiguous owned storage.
  using as_contig_owned_t = detail::StorageAs<contiguous_owned_t, T>;

 public:
  //==--- [constants] ------------------------------------------------------==//

  /// Defines if the type T is a StridableLayout type.
  static constexpr bool is_stridable_layout = true;
  /// Defines the type of the layout for T.
  static constexpr auto layout_kind         =
    detail::StorageLayoutKind<T>::value;
    /// True if the Layout is a LayoutKind::strided_view.
  static constexpr auto is_strided_view     =
    layout_kind == LayoutKind::strided_view;

  //==--- [traits] ---------------------------------------------------------==//

  /// Defines the type to use when storing in an iterator.
  using iter_storage_t   =
    std::conditional_t<is_strided_view, strided_view_t, contig_view_t>;
  /// Defines the type of the allocator for type T.
  using allocator_t      = typename iter_storage_t::allocator_t;
  /// Defines the value type for the layout.
  using value_t          = std::decay_t<T>;
  /// Defines the type T with owned storage for copying.
  using iter_copy_t      = typename as_contig_owned_t::type;
  /// Defines the type when referencing from an iterator.
  using iter_ref_t       = value_t;
  /// Defines the type of a const reference for an iterator.
  using iter_const_ref_t = const value_t;
  /// Defines the type of a pointer for an iterator.
  using iter_ptr_t       = PointerWrapper<value_t>;
  /// Defines thet type of a const pointer for an iterator.
  using iter_const_ptr_t = const iter_ptr_t;
};

} // namespace ripple

#endif // RIPPLE_STORAGE_LAYOUT_TRAITS_HPP

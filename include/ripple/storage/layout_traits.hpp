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
/// \file  storage_descriptor_traits.hpp
/// \brief This file defines a wrapper class to describe storage, and traits to
///        determine properties of the storage.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_LAYOUT_TRAITS_HPP
#define RIPPLE_STORAGE_LAYOUT_TRAITS_HPP

#include "detail/storage_traits_impl_.hpp"
#include "default_storage.hpp"
#include "storage_layout.hpp"

namespace ripple {

/// Defines layout traits for the the T. This is the default implementation for
/// the case that the type T does not implement the AutoLayout interface.
/// \tparam T             The type to get the layout traits for.
/// \tparam IsAutoLayable If the type is an AutoLayout type.
template <typename T, bool IsAutoLayable>
struct LayoutTraits {
 private:
  /// Defines the value type.
  using value_t = std::decay_t<T>;

  /// The AllocationTraits struct defines traits which can be used to determine
  /// properties of the allocation for the type T, for a speicific layout
  /// Layout.
  /// \tparam Layout The kind of the layout for the allocation.
  template <LayoutKind Layout>
  struct AllocationTraits { 
    /// Defines the type of the allocator for the data.
    using allocator_t = typename DefaultStorage<T>::allocator_t;
    /// Defines the type for a reference to the data. This is the view type.
    using ref_t       = value_t&;
    /// Defines the type for a constant reference to the data.
    using const_ref_t = const value_t&;
    /// Defines the type for a copy.
    using copy_t      = value_t;
  };

 public:
  /// Defines the type of the allocation traits for type T, with layout Layout.
  /// \tparam Layout The kind of the layout for the allocation traits. For
  ///         normal (types which do not implement the AutoLayout interface),
  ///         the Layout parameter does not have an effect.
  template <LayoutKind Layout>
  using alloc_traits_t = AllocationTraits<Layout>;
};

/// Defines layout traits for the the T when the type is auto layable.
/// \tparam T The type to get the layout traits for.
template <typename T>
struct LayoutTraits<T, true> {
 private:
  /// Defines the type T with strided storage.
  using strided_t    = typename detail::StorageAs<strided_view_t, T>::type;
  /// Defines the type T with contiguous storage.
  using contiguous_t = typename detail::StorageAs<contiguous_view_t, T>::type;
  /// Defines the type T with owned storage.
  using owned_t      = typename detail::StorageAs<contiguous_owned_t, T>::type;
  /// Defines the type of the descriptor for the type T.
  using descriptor_t = typename T::descriptor_t;

  /// The AllocationTraits struct defines traits which can be used to determine
  /// properties of the allocation for the type T, for a speicific layout
  /// Layout.
  /// \tparam Layout The kind of the layout for the allocation.
  template <LayoutKind Layout>
  struct AllocationTraits {
   private:
    /// Defines the type for allocation. For AutoLayout types, this will always
    /// use a view type, either strided or contiguous.
    using allocation_t = std::conditional_t<
      Layout == LayoutKind::strided_view,
      typename descriptor_t::strided_view_storage_t,
      typename descriptor_t::contig_view_storage_t
    >;

   public:
    /// Defines the type of the allocator for allocating type T types.
    using allocator_t = typename allocation_t::allocator_t;
    /// Defines the type to use when making a copy of T.
    using copy_t      = owned_t;
    /// Defines a type for a reference to type T.
    using ref_t       = std::conditional_t<
      Layout == LayoutKind::strided_view, strided_t, contiguous_t
    >;
    /// Definesa a type for a constant reference to type T.
    using const_ref_t = std::conditional_t<
      Layout == LayoutKind::strided_view,
      const strided_t, const contiguous_t
    >;
  };

 public:
  /// Defines the type of the allocation traits for type T.
  /// \tparam Layout The layout for the allocation traits.
  template <LayoutKind Layout>
  using alloc_traits_t = AllocationTraits<Layout>;
};

} // namespace ripple

#endif // RIPPLE_STORAGE_LAYOUT_TRAITS_HPP

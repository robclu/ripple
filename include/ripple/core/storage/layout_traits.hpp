//==--- ripple/core/storage/layout_traits.hpp -------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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

/**
 * Defines layout traits for the template type T. This implementation is for
 * when the data is owned by the type, other cases are specialized.
 *
 * \note This should be used as layout_traits_t<T>, since that aliases the
 *       correct specializtion.
 *
 * \tparam T                   The type to get the layout traits for.
 * \tparam PolyAndNonOwningData Condition for specialization.
 */
template <typename T, bool PolyAndNonOwningData>
struct LayoutTraits {
  // clang-format off
  /** Defines if the type T is a PolymorphicLayout type. */
  static constexpr bool is_polymorphic_layout = false;
  /** Defines the type of the layout for T. */
  static constexpr auto layout_kind           = LayoutKind::none;
  /** True if the Layout is a strided view layout kind. */
  static constexpr auto is_strided_view       = false;

  /** Defines the value type of T. */
  using Value        = std::decay_t<T>;
  /** Defines the type of the allocator for type T. */
  using Allocator    = typename DefaultStorage<T>::Allocator;
  /** Defines the type for making a copy of T. */
  using IterCopy     = Value;
  /** Defines the type to use when storing in an iterator. */
  using IterStorage  = Value*;
  /** Defines the type when referencing from an iterator. */
  using IterRef      = Value&;
  /** Defines the type of a const reference for an iterator. */
  using IterConstRef = const Value&;
  /** Defines the type of a pointer for an iterator. */
  using IterPtr      = Value*;
  /** Defines the type of a const pointer for an iterator. */
  using IterConstPtr = const Value*;
  /** Defines the type of the raw pointer to the data. */
  using RawPtr       = Value*;
  /** Defines the type of the const raw pointer to the data. */
  using ConstRawPtr  = const Value*;
  // clang-format on
};

/**
 * Specialization of the layout traits for caset that the template type T
 * implements the PolymorphicLayout interface, and when the layout kind for the
 * type is a view type, which means that it *does not* own the data, and the
 * data is therefore allocated somewhere else the the storage points to it.
 *
 * \note This should be used as layout_traits_t<T>, since that aliases the
 *       correct specializtion.
 *
 * \tparam T The type to get the layout traits for.
 */
template <typename T>
struct LayoutTraits<T, true> {
 private:
  // clang-format off
  /** Defines the type of the descriptor for the type T. */
  using Descriptor    = typename T::Descriptor;
  /** Defines the type for strided view storage. */
  using StridedView   = typename Descriptor::StridedView;
  /** Defines the type for contiguous view storage. */
  using ContigView    = typename Descriptor::ContigView;
  /** Defines the type T with contiguous owned storage. */
  using AsContigOwned = detail::StorageAs<ContiguousOwned, T>;

 public:
  /*==--- [constants] ------------------------------------------------------==*/

  /** Defines if the type T is a PolymorphicLayout type. */
  static constexpr bool is_polymorphic_layout = true;

  /** Defines the type of the layout for T. */
  static constexpr auto layout_kind = detail::StorageLayoutKind<T>::value;

  /** True if the Layout is a strided view kind. */
  static constexpr auto is_strided_view =
    layout_kind == LayoutKind::strided_view;

  /** True if the Layout is a contiguous view kind */
  static constexpr auto is_contiguous_view =
    layout_kind == LayoutKind::contiguous_view;

  /** Defines the type to use when storing in an iterator. */
  using IterStorage = std::conditional_t<
    is_strided_view, StridedView, ContigView>;

  /** Defines the type of the allocator for type T. */
  using Allocator    = typename IterStorage::Allocator;
  /** Defines the value type for the layout. */
  using Value        = std::decay_t<T>;
  /** Defines the type T with owned storage for copying. */
  using IterCopy     = typename AsContigOwned::type;
  /** Defines the type when referencing from an iterator. */
  using IterRef      = Value;
  /** Defines the type of a const reference for an iterator. */
  using IterConstRef = const Value;
  /** Defines the type of a pointer for an iterator. */
  using IterPtr      = PointerWrapper<Value>;
  /** Defines thet type of a const pointer for an iterator. */
  using IterConstPtr = const IterPtr;
  /** Defines the type of the raw pointer to the data. */
  using RawPtr       = void*;
  /** Defines the type of the const raw pointer to the data. */
  using ConstRawPtr  = const void*;
  // clang-format off
};

} // namespace ripple

#endif // RIPPLE_STORAGE_LAYOUT_TRAITS_HPP

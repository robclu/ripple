//==--- ripple/core/storage/storage_traits.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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
#include "layout_traits.hpp"

namespace ripple {

/*==--- [forward declarations] ---------------------------------------------==*/

/**
 * A type for view storage for the types defined by the template arguments.
 *
 * The view is created as if the Ts types are stored contiguously (i.e AoS),
 * where the view points to an element in large array where the components of
 * the array look like classes with Ts as their members. Again, this is Aos
 * layout (i.e, how data would usually be allocated).
 *
 * For example, given `Ts = int, float, double', this would create storage
 * as follows:
 *
 * ~~~
 * .---------------------------------------------------.
 * |      element 0       | ... |     element N -1     |
 * .---------------------------------------------------.
 * | int | float | double | ... | int | float | double |
 * '---------------------------------------------------'
 * ~~~
 *
 *
 * Since this is a view type storage, it does not own the underlying data, it
 * just references data which has been allocated with the appropriate layout.
 *
 * It provides access to the types via the StorageAccessor interface.
 *
 * If any of the Ts are Vector<T, V> types, then this will provide
 * storage such each of the V components of the element are contiguous
 * (i.e as Aos), and also contiguous with the types on either side of the
 * Vector in the Ts pack.
 *
 * To allocate multiple ContiguousStorageiView elements, the
 * Allocator type should be used to determine the memory
 * requirement, and then a ContiguousStorageView can be created through the
 * allocator, as also offset through the allocator.
 *
 * Currently, this requires that the Ts be ordered in descending alignment
 * size, as there is no functionality to sort the types. If they are not
 * ordered, this will still allocate and align the types correctly, however,
 * the alignment might result in padding which could add a lot of unnecessary
 * data. This behaviour mimics what the compiler would do, so following best
 * practices and ensuring that larger types are placed first will ensure the
 * minimum size of each element.
 *
 * Sorting can't be added because it will cause the `get` interface to break,
 * since it relies on knowing the order of the types, which are specified by the
 * user. Sorting would mean that the user would not know the order, and would
 * have to provide the type as the index to get, which is not so nice.
 *
 * \tparam Ts The types to create storage for.
 */
template <typename... Ts>
class ContiguousStorageView;

/**
 * A class to create owned storage for the types defined by the template
 * arguments.
 *
 * It creates the storage as if the Ts types are stored contiguously (i.e AoS).
 * This will create the data for the storage statically, as is done for a
 * normal class.
 *
 * This class owns the storage data.
 *
 * Returning this from an array, or setting it from other StorageAccessor types
 * does not allow that data to be modified, and therefore its use case is for
 * data modification (i.e local variables), or when the storage interface is
 * desired but an implementation which mimics traditional structs is required.
 *
 *
 * It provides access to the types via the StorageAccessor interface.
 *
 * If any of the Ts are Vector<T, V> types, then this will provide
 * storage such that each of the V components of the element are contiguous
 * (i.e as AoS), and also contiguous with the types on either side of the
 * Vector in the Ts pack.
 *
 * This class does not have an allocator inteface, and is not intended to be
 * allocated, but rather reutned by data containers which use
 * ContiguousStorageView/StridedStorageView when a local copy is required, for
 * example, when deferencing shared memory data which uses StridedStorage.
 *
 * Currently, this requires that the Ts be ordered in descending alignment
 * size, as there is no functionality to sort the types. If they are not
 * ordered, this will still allocate and align the types correctly, however,
 * the alignment might result in padding which could add a lot of unnecessary
 * data. This behaviour mimics what the compiler would do, so following best
 * practices and ensuring that larger types are placed first will ensure the
 * minimum size of each element.
 *
 * Sorting can't be added because it will cause the `get` interface to break,
 * since it relies on knowing the order of the types, which are specified by the
 * user. Sorting would mean that the user would not know the order, and would
 * have to provide the type as the index to get, which is not so nice.
 *
 * \tparam Ts The types to create storage for.
 */
template <typename... Ts>
class OwnedStorage;

/**
 * A view storage class for the types defined by the template arguments.
 *
 * It creates the view into storage as if the Ts types are stored strided (i.e
 * SoA), where the view points to an element in which is effectively multiple
 * arrays, each of which is one of the Ts types. Again, this is SoA layout.
 *
 * Since this is a view type storage, it does not own the underlying data, it
 * just references data which has been allocated with the appropriate layout.
 *
 * It provides access to the types via the StorageAccessor interface.
 *
 * If any of the Ts are Vector<T, V> types, then this will provide
 * storage such each of the V components of the element are strided
 * (i.e as SoA).
 *
 * To allocate multiple StridedStorageiView elements, the
 * Allocator type should be used to determine the memory requirement, and
 * then a StridedStorageView can be created through the allocator, as also
 * offset through the allocator.
 *
 * Currently, this requires that the Ts be ordered in descending alignment
 * size, as there is no functionality to sort the types. This may be added in
 * the future, but the compile time overhead of performing a sort of the types
 * based on their required alignments is significant.
 *
 * \tparam Ts The types to create a storage view for.
 */
template <typename... Ts>
class StridedStorageView;

/**
 * The PolymorphicLayout class defines a class which can be used as an empty
 * base class to define that the layout of the data for the class is
 * polymorphic.
 *
 * \tparam Impl The implementation type with a polymorphic layout.
 */
template <typename Impl>
struct PolymorphicLayout;

/**
 * The StorageAccessor types defines an interface for all classes which
 * access storage using compile time indices, where the compile time indices
 * are used to access the Ith type in the storage for N type, and for storage
 * where the Ith type is an array type, also any of the J elements.
 * \tparam Impl The implementation of the interface.
 */
template <typename Impl>
struct StorageAccessor;

/*==--- [traits] -----------------------------------------------------------==*/

/**
 * Determines if a type is a polymorphic layout type.
 * \tparam T The type to determine if implements the interface.
 * \return true if the type implements the polymorphic layout interface.
 */
template <typename T>
static constexpr auto is_polymorphic_layout_v =
  std::is_base_of_v<PolymorphicLayout<std::decay_t<T>>, std::decay_t<T>>;

/**
 * Determines if a type is a StorageLayout type.
 * \tparam T The type to determine if is a StoageLayout type.
 * \return true if the type is a storage layout type.
 */
template <typename T>
static constexpr auto is_storage_layout_v =
  detail::IsStorageLayout<std::decay_t<T>>::value;

/**
 * Determines if a type is a StorageAccessor.
 * \param T The type to check if is a storage accessor.
 * \return true if the type is a storage accessor type.
 */
template <typename T>
static constexpr auto is_storage_accessor_v =
  std::is_base_of_v<StorageAccessor<std::decay_t<T>>, std::decay_t<T>>;

/**
 * Determines if a type has a StorageLayout type as a template parameter.
 *
 * \note This will not work if the class has template parameters which are not
 *       types (i.e non-type template parameters). If this is required, the
 *       numeric template parameters can be wrapped in the Num or Int64 classes
 *       before calling this.
 *
 * \tparam T The type to determine if has a storage layout parameter.
 * \return true if the type has a template parameter which is a storage layout.
 */
template <typename T>
static constexpr auto has_storage_layout_v =
  detail::HasStorageLayout<std::decay_t<T>>::value;

/**
 * Determines the type of the storage layout for a type, if it has one.
 * \tparam T The type to get the storage layout for.
 * \return The kind of the storage layout if the type has one.
 */
template <typename T>
static constexpr auto storage_layout_kind_v =
  detail::StorageLayoutKind<T>::value;

/**
 * Determines if the storage layout kind of the type is the same as the
 * comparison kind.
 * \tparam T    The types whose storage layout to get.
 * \tparam Kind The kind to check for.
 * \return true if the Kind matches the layout kind of the type T.
 */
template <typename T, typename Kind>
static constexpr auto storage_layout_kind_is_v =
  storage_layout_kind_v<std::decay_t<T>> == Kind::value;

/*==-- [aliases] -----------------------------------------------------------==*/

/**
 * Alias for storage layout traits.
 * \tparam T The type to get the traits for.
 */
template <typename T>
using layout_traits_t = LayoutTraits<
  std::decay_t<T>,
  is_polymorphic_layout_v<std::decay_t<T>> &&
    !storage_layout_kind_is_v<T, ContiguousOwned>>;

/**
 * Defines the type T as a contiguous owned type, if it is not already. If the
 * type T is not a polymorphic layout type then this will just create an alias
 * to T, otherwise the layout will be converted.
 * \tparam T The type to get as a contiguous owned type.
 */
template <typename T>
using as_contiguous_owned_t =
  typename detail::StorageAs<ContiguousOwned, std::decay_t<T>>::Type;

/**
 * Defines the type T as a contiguous view type, if it is not already. If the
 * type T is not a polymorphic layout type then this will just create an alias
 * to T, otherwise the layout will be converted.
 * \tparam T The type to get as a contiguous owned type.
 */
template <typename T>
using as_contiguous_view_t =
  typename detail::StorageAs<ContiguousView, std::decay_t<T>>::Type;

/**
 * Defines the type T as a strided view type, if it is not already. If the
 * type T is not a polymorphic layout type this will just create an alias to T
 * otherwise the layout will be converted.
 * \tparam T The type to get as a contiguous owned type.
 */
template <typename T>
using as_strided_view_t =
  typename detail::StorageAs<StridedView, std::decay_t<T>>::Type;

/*==--- [overloading] ------------------------------------------------------==*/

/**
 * The PolyLayoutOverloader struct can be used to overload functions for types
 * which are PolymorphicLayout.
 * \tparam IsPolymorphicLayout If the class is a polymorphic layout type.
 */
template <bool IsPolymorphicLayout>
struct PolyLayoutOverloader {};

// clang-format off
/** Defines an alias for an overload type for polymorphic layout types. */
using PolyLayoutOverload    = PolyLayoutOverloader<true>;
/** Defines an alias for an overload type for non polymorphic layout types. */
using NonPolyLayoutOverload = PolyLayoutOverloader<false>;
// clang-format off

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

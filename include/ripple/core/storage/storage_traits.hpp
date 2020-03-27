//==--- ripple/core/storage/storage_traits.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
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

//==--- [forward declarations] ---------------------------------------------==//

/// This creates a view storage for the Ts types defined by the arguments, and
/// it creates the view as if the Ts types are stored contiguously (i.e AoS),
/// where the view points to an element in large array where the components of
/// the array look like classes with Ts as their members. Again, this is Aos
/// layout (i.e, how data would usually be allocated).
///
/// Since this is a view type storage, it does not own the underlying data, it
/// just references data which has been allocated with the appropriate layout.
///
/// It provides access to the types via the StorageAccessor interface.
///
/// If any of the Ts are StorageElement<T, V> types, then this will provide
/// storage such each of the V components of the element are contiguous
/// (i.e as Aos), and also contiguous with the types on either side of the
/// StorageElement in the Ts pack.
///
/// To allocate multiple ContiguousStorageiView elements, the publicly
/// accessible allocator_t type should be used to determine the memory
/// requirement, and then a ContiguousStorageView can be created through the
/// allocator, as also offset through the allocator.
///
/// Currently, this requires that the Ts be ordered in descending alignment
/// size, as there is no functionality to sort the types. If they are not
/// ordered, this will still allocate and align the types correctly, however,
/// the alignment might result in padding which could add a lot of unnecessary
/// data. This sorting may be added in the future, but the compile time overhead
/// of performing a sort of the type based on their required alignments is
/// significant.
///
/// \tparam Ts The types to create storage for.
template <typename... Ts> class ContiguousStorageView;

/// This creates an owned storage for the Ts types defined by the arguments, and
/// it creates the storage as if the Ts types are stored contiguously (i.e AoS).
/// This will create the data for the storage statically, as is done for a
/// normal class.
///
/// Since this is an owned type storage, and hence owns the underlying data.
/// Returning this from an array, or setting it from other StorageAccessor types
/// does not allow that data to be modified, and therefore it's use is when the
/// data need to be modified locally, where contiguous storage is required (i.e
/// in the cache or registers on the GPU).
///
/// It provides access to the types via the StorageAccessor interface.
///
/// If any of the Ts are StorageElement<T, V> types, then this will provide
/// storage such each of the V components of the element are contiguous
/// (i.e as AoS), and also contiguous with the types on either side of the
/// StorageElement in the Ts pack.
///
/// This class does not have an allocator inteface, and is not intended to be
/// allocated, but rather reutned by data containers which use
/// ContiguousStorageView/StridedStorageView when a local copy is required.
///
/// Currently, this requires that the Ts be ordered in descending alignment
/// size, as there is no functionality to sort the types. If they are not
/// ordered, this will still allocate and align the types correctly, however,
/// the alignment might result in padding which could add a lot of unnecessary
/// data. This sorting may be added in the future, but the compile time overhead
/// of performing a sort of the type based on their required alignments is
/// significant.
///
/// \tparam Ts The types to create storage for.
template <typename... Ts> class OwnedStorage;

/// This creates a view storage for the Ts types defined by the arguments, and
/// it creates the view as if the Ts types are stored strided (i.e SoA), where
/// the view points to an element in which is effectively multiple arrays, each
/// of which is one of the Ts types. Again, this is SoA layout.
///
/// Since this is a view type storage, it does now own the underlying data, it
/// just references data which has been allocated with the appropriate layout.
///
/// It provides access to the types via the StorageAccessor interface.
///
/// If any of the Ts are StorageElement<T, V> types, then this will provide
/// storage such each of the V components of the element are strided 
/// (i.e as SoA).
///
/// To allocate multiple StridedStorageiView elements, the publicly accessible
/// allocator_t type should be used to determine the memory requirement, and
/// then a StridedStorageView can be created through the allocator, as also
/// offset through the allocator.
///
/// Currently, this requires that the Ts be ordered in descending alignment
/// size, as there is no functionality to sort the types. This may be added in
/// the future, but the compile time overhead of performing a sort of the types
/// based on their required alignments is significant.
///
/// \tparam Ts The types to create a storage view for.
template <typename... Ts> class StridedStorageView;

//==--- [interfaces] -------------------------------------------------------==//

/// The StridableLayout class defines a static interface for classes to implement
/// for which is might be beneficial to allocate the class data in a strided
/// layout -- essentially any class which might be used for processing on either
/// the GPU or using AVX -- which is more performant.
///
/// Inheriting this static interface will allow any containers which can use the
/// strided allocators to do so where appropriate.
///
/// \tparam Impl The implementation of the interface.
template <typename Impl> struct StridableLayout;

/// The StorageAccessor struct defines an interface for all classes which
/// access storage using compile time indices, where the compile time indices
/// are used to access the Ith type in the storage, and for storage where the
/// Ith type is an array type, the Jth element.
/// \tparam Impl The implementation of the interface.
template <typename Impl> struct StorageAccessor;

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type I implements the StridableLayout interface.
/// \tparam T The type to determine if implements the interface.
template <typename T>
static constexpr auto is_stridable_layout_v 
  = std::is_base_of_v<StridableLayout<std::decay_t<T>>, std::decay_t<T>>;

/// Returns true if the type T is a StorageLayout type, otherwise returns false.
/// \tparam T The type to determine if is a StoageLayout type.
template <typename T>
static constexpr auto is_storage_layout_v = 
  detail::IsStorageLayout<std::decay_t<T>>::value;

/// Returns true if the type T has template parameters and one of the template
/// parameters is a StorageLayout type.
///
/// \note This will not work if the class has template parameters which are not
///       types. If this is required, the numeric template parameters can be
///       wrapped in the Num or Int64 classes.
///
/// \tparam T The type to determine if has a storage layout parameter.
template <typename T>
static constexpr auto has_storage_layout_v =
  detail::HasStorageLayout<std::decay_t<T>>::value;

/// Returns the type of storage layout for the type T, if is has one.
/// \tparam T The type to get the storage layout for.
template <typename T>
static constexpr auto storage_layout_kind_v =
  detail::StorageLayoutKind<T>::value;

/// Returns true if the layout kind of T is the same as Kind.
/// \tparam T    The types whose storage layout to get.
/// \tparam Kind The kind to check for.
template <typename T, typename Kind>
static constexpr auto storage_layout_kind_is_v =
  storage_layout_kind_v<std::decay_t<T>> == Kind::value;

//==-- [aliases] -----------------------------------------------------------==//

/// Alias for storage layout traits.
/// \tparam T The type to get the traits for.
template <typename T>
using layout_traits_t = LayoutTraits< std::decay_t<T>, 
  is_stridable_layout_v<std::decay_t<T>> &&
  !storage_layout_kind_is_v<T, contiguous_owned_t>
>;

/// Returns the type T as a contiguous owned type, if it is not already. If the
/// type T is not stridable, this will just create an alias to T.
/// \tparam T The type to get as a contiguous owned type.
template <typename T>
using as_contiguous_owned_t = 
  typename detail::StorageAs<contiguous_owned_t, std::decay_t<T>>::type;

/// Returns the type T as a contiguous view type, if it is not already. If the
/// type T is not stridable, this will just create an alias to T.
/// \tparam T The type to get as a contiguous owned type.
template <typename T>
using as_contiguous_view_t = 
  typename detail::StorageAs<contiguous_view_t, std::decay_t<T>>::type;

//==--- [enables] ----------------------------------------------------------==//

/// Define a valid type if the type T is a StorageElement, otherwise does not
/// define a valid type.
/// \tparam T The type to base the enable on.
template <typename T>
using storage_element_enable_t = std::enable_if_t<is_storage_element_v<T>, int>;

/// Define a valid type if the type T is not a StorageElement, otherwise does
/// not define a valid type.
/// \tparam T The type to base the enable on.
template <typename T>
using non_storage_element_enable_t =
  std::enable_if_t<!is_storage_element_v<T>, int>;

//==--- [overloading] ------------------------------------------------------==//

/// The StridableOverloader struct can be used to overload functions for types
/// which are Stridable.
/// \tparam IsStridable If the class is stridable.
template <bool IsStridable> struct StridableOverloader {};

/// Defines an alias for an overload type for stridable types.
using stridable_overload_t     = StridableOverloader<true>;
/// Defines an alias for an overload type for non stridable types.
using non_stridable_overload_t = StridableOverloader<false>;

//==--- [constants] --------------------------------------------------------==//

/// Defines the number of bytes to allocate to avoid false sharing. This is 128b
/// rather than 64b because in the Intel Optimization guide, chapter 2.1.5.4,
/// the prefetcher likes to keep pairs of cache lines in the L2 cache. 
static constexpr auto false_sharing_size = 128;

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

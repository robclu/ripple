//==--- ripple/storage/storage_traits.hpp ------------------ -*- C++ -*- ---==//
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
#include "storage_element_traits.hpp"

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

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T is a StorageLayout type, otherwise returns false.
/// \tparam T The type to determine if is a StoageLayout type.
template <typename T>
static constexpr auto is_storage_layout_v = 
  detail::IsStorageLayout<std::decay_t<T>>::value;

/// Returns true if the type T is a StorageElement type, otherwise returns
/// false.
/// \tparam T The type to determine if is a StoageElement type.
template <typename T>
static constexpr auto is_storage_element_v = 
  detail::IsStorageElement<std::decay_t<T>>::value;

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

//==-- [aliases] -----------------------------------------------------------==//

/// Defines an alias for storage which is laid out contiguously.
using contiguous_layout_t = StorageLayout<LayoutKind::contiguous>;
/// Defines an alias for storage which is laid out strided.
using strided_layout_t    = StorageLayout<LayoutKind::strided>;

/// Alias for storage element traits.
/// \tparam T The type to get the traits for.
template <typename T>
using storage_element_traits_t = StorageElementTraits<std::decay_t<T>>;

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

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

//==--- ripple/storage/detail/storage_traits_impl_.hpp ----- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_traits_impl_.hpp
/// \brief This file defines implementation details for storage traits.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_DETAIL_STORAGE_TRAITS_IMPL_HPP
#define RIPPLE_STORAGE_DETAIL_STORAGE_TRAITS_IMPL_HPP

#include "../storage_element_traits.hpp"
#include "../storage_layout.hpp"
#include <ripple/utility/type_traits.hpp>

namespace ripple {
namespace detail {

//==--- [is storage layout] ------------------------------------------------==//

/// Defines a class to determine of the type T is a layout type.
/// \tparam T The type to determine if is a storage layout type.
template <typename T>
struct IsStorageLayout : std::false_type {
  /// Defines that the type is not a storage layout type.
  static constexpr auto value = false;
};

/// Specialization for the case of the IsStorageLayout struct for the case the
/// type to check is a StorageLayout.
/// \tparam Layout The kind of the layout for the storage.
template <LayoutKind Layout>
struct IsStorageLayout<StorageLayout<Layout>> : std::true_type {
  /// Defines that the type is a storage layout type.
  static constexpr auto value = true;
};

//==--- [is storage element] -----------------------------------------------==//

/// Defines a class to determine of the type T is a storage element type.
/// \tparam T The type to determine if is a storage element type.
template <typename T>
struct IsStorageElement : std::false_type {
  /// Defines that the type is not a storage element type.
  static constexpr auto value = false;
};

/// Specialization for the case of the IsStorageElement struct for the case the
/// type to check is a StorageElement.
/// \tparam T     The type for the element.
/// \tparam Value The number of values for the type.
template <typename T, std::size_t Values>
struct IsStorageElement<StorageElement<T, Values>> : std::true_type {
  /// Defines that the type is a storage element type.
  static constexpr auto value = true;
};

//==--- [has storage layout] -----------------------------------------------==//

/// Defines a struct to determine if the type T has a storage layout type
/// template parameter.
/// \tparam T The type to determine if has a storage layout parameter.
template <typename T>
struct HasStorageLayout {
  /// Returns that the type does not have a storage layout paramter.
  static constexpr auto value = false;
};

/// Specialization for the case that the type has template parameters.
/// \tparam T  The type to determine if has a storage layout parameter.
/// \tparam Ts The types for the type T.
template <template <class...> typename T, typename... Ts>
struct HasStorageLayout<T<Ts...>> {
  /// Returns that the type does not have a storage layout paramter.
  static constexpr auto value = std::disjunction_v<IsStorageLayout<Ts>...>;
};

//==--- [storage as] -------------------------------------------------------==//

/// Returns the type T, with the storage type of Storage, if T has templates
/// paramters which can be replaced by a StorageLayout type.
///
/// This is the default case that T does not have any types which can be
/// replaced witha StorageLayout type.
///
/// \tparam Storage The storage type for T.
/// \tparam T       The type to specify storage for.
template <typename Storage, typename T>
struct StorageAs {
  /// Defines the type with the specified storage.
  using type = T;
};

namespace detail {

/// Implementation utility for StorageAs.
/// \tparam Storage The desired storage layout.
/// \tparam T1      A type list of untried types.
/// \tparam T2      A type list of tried types.
template <typename Storage, typename T1, typename T2>
struct StorageAsImpl;

/// Specialization to compare the type T against the Storage and replace it if
/// necessary.
/// \tparam Storage The desired storage layout.
/// \tparam T       The type to try and replace with Storage.
/// \tparam Ts      A type list of untried types.
/// \tparam Us      A type list of tried types.
template <typename Storage, typename T, typename... Ts, typename... Us>
struct StorageAsImpl<Storage, std::tuple<T, Ts...>, std::tuple<Us...>> {
  /// Returns the list of types for the type to replace with Storage.
  /// If T is a StorageLayout type, replace T with Storage, and return,
  /// otherwise type the rest of the untried types.
  using type = std::conditional_t<
    IsStorageLayout<T>::value,
    std::tuple<Us..., Storage, Ts...>,
    typename StorageAsImpl<
      Storage, std::tuple<Ts...>, std::tuple<Us..., T>
    >::type
  >;
};

/// Specialization for the case that all types have been tried, but non matched.
/// \tparam Storage The desired storage type.
/// \tparam Ts      All the tried types.
template <typename S, typename... Ts>
struct StorageAsImpl<S, std::tuple<>, std::tuple<Ts...>> {
  /// Returns the original list of types.
  using type = std::tuple<Ts...>;
};

} // namespace detail

/// Returns the type T with the same types Ts, except if any of the types Ts are
/// StorageLayout types, then that type will be replaced by Storage.
/// \tparam Storage The desired storage type.
/// \tparam T       The type to specify the storage type for.
/// \tparam Ts      The template types  for T.
template <
  typename                        Storage,
  template <typename...> typename T,
  typename...                     Ts
>
struct StorageAs<Storage, T<Ts...>> {
 private:
  /// Defines the tuple type with the replaced storage type.
  using type_list_t = typename detail::StorageAsImpl<
    Storage, std::tuple<Ts...>, std::tuple<>
  >::type;

  /// Makes the storage type for the arguments Us.
  /// \tparam Us The arguments to make the storage type with.
  template <typename... Us> struct MakeStorageType;

  /// Specialization for when the types are contained in a tuple.
  /// \tparam Us The types to make the storage type with.
  template <typename... Us>
  struct MakeStorageType<std::tuple<Us...>> {
    /// Defines the type of the storage type.
    using type = T<Us...>;
  };

 public:
  /// Defines the type of the type T with a StorageElement type replaced by
  /// Storage.
  using type = typename MakeStorageType<type_list_t>::type;
};

//==--- [storage layout kind] ----------------------------------------------==//

namespace {
/// Declaration of a class to determine the type of a StorageLayout from a list
/// of types.
/// \tparam Ts The types to find the StorageLayout type in.
template <typename... Ts>
struct StorageLayoutKindImpl;

/// Specialization for the case that there are multiple types in the type list.
/// \tparam T   The type to compare.
/// \tparam Ts  The rest of the types to compare.
template <typename T, typename... Ts>
struct StorageLayoutKindImpl<T, Ts...> {
  /// Defines the type of the StorageLayout type.
  using type = std::conditional_t<IsStorageLayout<T>::value,
    T, typename StorageLayoutKindImpl<Ts...>::type
  >;

  /// Defines the value of the storage layout kind.
  static constexpr auto value = type::value;
};

/// Specialization for the base case where there are no types to compare. This
/// just returns that there is no storage layout specified.
template <>
struct StorageLayoutKindImpl<> {
  /// Defines the type of the storage layout.
  using type = StorageLayout<LayoutKind::none>;
};

} // namespace annon

/// Determines the LayoutKind for the type T, if it has a StorageLayout<> as
/// one of its template parameters.
/// \tparam T The type to find the storage layout kind for.
template <typename T>
struct StorageLayoutKind {
  /// The value of the LayoutKind for type T.
  static constexpr auto value = StorageLayoutKindImpl<T>::value;
};

/// Specialization for the case that T is a class with template parameters.
/// \tparam T The type to find the LayoutKind of.
/// \tparam Ts The template types for T.
template <template <typename...> typename T, typename... Ts>
struct StorageLayoutKind<T<Ts...>> {
  /// Defines the value of the LayoutKind for T.
  static constexpr auto value = StorageLayoutKindImpl<Ts...>::value;
};

}} // namespace ripple::detail

#endif // RIPPLE_STORAGE_STORAGE_TRAITS_HPP

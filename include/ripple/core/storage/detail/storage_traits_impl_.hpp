//==--- ripple/core/storage/detail/storage_traits_impl_.hpp  -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_traits_impl_.hpp
/// \brief This file defines implementation details for storage traits.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_DETAIL_STORAGE_TRAITS_IMPL__HPP
#define RIPPLE_STORAGE_DETAIL_STORAGE_TRAITS_IMPL__HPP

#include "../storage_element_traits.hpp"
#include <ripple/core/utility/type_traits.hpp>

namespace ripple::detail {

/*==--- [storage as] -------------------------------------------------------==*/

/**
 * Returns the type T, with the storage type of Storage, if T has templates
 * parameters which can be replaced by a StorageLayout type.
 *
 * This is the default case that T does not have any types which can be
 * replaced witha StorageLayout type.
 *
 * \tparam Storage The storage type for T.
 * \tparam T       The type to specify storage for.
 */
template <typename Storage, typename T>
struct StorageAs {
  /** Defines the type with the specified storage. */
  using Type = T;
};

/**
 * Implementation utility for StorageAs.
 * \tparam Storage The desired storage layout.
 * \tparam T1      A type list of untried types.
 * \tparam T2      A type list of tried types.
 */
template <typename Storage, typename T1, typename T2>
struct StorageAsImpl;

/**
 * Specialization to compare the type T against the Storage and replace it if
 * necessary.
 * \tparam Storage The desired storage layout.
 * \tparam T       The type to try and replace with Storage.
 * \tparam Ts      A type list of untried types.
 * \tparam Us      A type list of tried types.
 */
template <typename Storage, typename T, typename... Ts, typename... Us>
struct StorageAsImpl<Storage, std::tuple<T, Ts...>, std::tuple<Us...>> {
  /**
   * Defines the list of types for the type to replace with Storage.
   * If T is a StorageLayout type, replaces T with Storage, otherwise
   * otherwise appends T to the list of checked types and continues.
   */
  using Type = std::conditional_t<
    IsStorageLayout<T>::value,
    std::tuple<Us..., Storage, Ts...>,
    typename StorageAsImpl<Storage, std::tuple<Ts...>, std::tuple<Us..., T>>::
      Type>;
};

/**
 * Specialization for the case that all types have been tried, but non matched.
 * \tparam Storage The desired storage type.
 * \tparam Ts      All the tried type.
 */
template <typename Storage, typename... Ts>
struct StorageAsImpl<Storage, std::tuple<>, std::tuple<Ts...>> {
  /** Defines the original list of types. */
  using Type = std::tuple<Ts...>;
};

/**
 * Returns the type T with the same types Ts, except if any of the types Ts are
 * StorageLayout types, then those types will be replaced by Storage.
 * \tparam Storage The desired storage type.
 * \tparam T       The type to specify the storage type for.
 * \tparam Ts      The template types  for T.
 */
template <typename Storage, template <typename...> typename T, typename... Ts>
struct StorageAs<Storage, T<Ts...>> {
 private:
  /** Defines the tuple type with the replaced storage type. */
  using TypeList = typename detail::
    StorageAsImpl<Storage, std::tuple<Ts...>, std::tuple<>>::Type;

  /**
   * Makes the storage type for the arguments Us.
   * \tparam Us The arguments to make the storage type with.
   */
  template <typename... Us>
  struct MakeStorageType;

  /**
   * Specialization for when the types are contained in a tuple.
   * \tparam Us The types to make the storage type with.
   */
  template <typename... Us>
  struct MakeStorageType<std::tuple<Us...>> {
    /** Defines the type of the storage type. */
    using Type = T<Us...>;
  };

 public:
  /**
   * Defines the type of the type T with a StorageElement type replaced by
   * Storage.
   */
  using Type = typename MakeStorageType<TypeList>::Type;
};

/*==--- [storage layout kind] ----------------------------------------------==*/

namespace {

/**
 * Declaration of a class to determine the type of a StorageLayout from a list
 * of types.
 * \tparam Ts The types to find the StorageLayout type in.
 */
template <typename... Ts>
struct StorageLayoutKindImpl;

/**
 * Specialization for the case that there are multiple types in the type list.
 * \tparam T   The type to compare.
 * \tparam Ts  The rest of the types to compare.
 */
template <typename T, typename... Ts>
struct StorageLayoutKindImpl<T, Ts...> {
  /** Defines the type of the StorageLayout type. */
  using Type = std::conditional_t<
    IsStorageLayout<T>::value,
    T,
    typename StorageLayoutKindImpl<Ts...>::Type>;

  /** Defines the value of the storage layout kind. */
  static constexpr auto value = Type::value;
};

/**
 * Specialization for the base case where there are no types to compare. This
 * just returns that there is no storage layout specified.
 */
template <>
struct StorageLayoutKindImpl<> {
  /** Defines the type of the storage layout. */
  using Type = StorageLayout<LayoutKind::none>;
};

} // namespace

/**
 * Determines the LayoutKind for the type T, if it has a StorageLayout<> as
 * one of its template parameters.
 * \tparam T The type to find the storage layout kind for.
 */
template <typename T>
struct StorageLayoutKind {
  /** The value of the LayoutKind for type T. */
  static constexpr auto value = StorageLayoutKindImpl<T>::value;
};

/**
 * Specialization for the case that T is a class with template parameters.
 * \tparam T The type to find the LayoutKind of.
 * \tparam Ts The template types for T.
 */
template <template <typename...> typename T, typename... Ts>
struct StorageLayoutKind<T<Ts...>> {
  /** Defines the value of the LayoutKind for T. */
  static constexpr auto value = StorageLayoutKindImpl<Ts...>::value;
};

} // namespace ripple::detail

#endif // RIPPLE_STORAGE_DETAIL_STORAGE_TRAITS_IMPL__HPP

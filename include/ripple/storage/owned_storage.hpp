//==--- ripple/storage/owned_storage.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  owned_storage.hpp
/// \brief This file implements a storage class which owns it storage.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_OWNED_STORAGE_HPP
#define RIPPLE_STORAGE_OWNED_STORAGE_HPP

#include "storage_traits.hpp"
#include <ripple/utility/type_traits.hpp>

namespace ripple {

/// This creates storage for the Ts types defined by the arguments, and it
/// creates the storage as pointers to arrays of each of the types Ts. It then
/// provides access to the types via the `get<>` method, to get a reference to
/// the appropriate type.
///
/// The OwnedStorage class defines a class which has storage for the types
/// defined by the Ts types, which is owned by the class itself. The data for
/// the types is stored contiguously.
///
/// Currently, this requires that the Ts be ordered in descending alignment
/// size, as there is no functionality to sort the types, otherwise padding will
/// be added to align the types correctly.
///
/// \tparam Ts The types to create storage for.
template <typename... Ts>
class OwnedStorage {
  /// Defines the data type for the buffer.
  using buffer_t  = char;
  /// Defines the type of the storage.
  using storage_t = OwnedStorage<Ts...>;

  //==--- [traits] ---------------------------------------------------------==//

  /// Gets the value type of the storage element traits for a type.
  /// \tparam T The type to get the storage element traits for.
  template <typename T>
  using element_value_t = typename storage_element_traits_t<T>::value_t;

  //==--- [constants] ------------------------------------------------------==//

  /// Defines the number of different types.
  static constexpr auto num_types = sizeof...(Ts);

  /// Gets the numbber of components for the storage element.
  /// \tparam T The type to get the size of.
  template <typename T>
  static constexpr auto element_components_v =
    storage_element_traits_t<T>::num_elements;

  /// Defines the sizes of each of the types.
  static constexpr std::size_t byte_sizes[num_types] = {
    storage_element_traits_t<Ts>::byte_size...
  };
  /// Defines the alignment sizes of each of the types.
  static constexpr std::size_t align_sizes[num_types] = {
    storage_element_traits_t<Ts>::align_size...
  };

  /// Returns the effective byte size of all elements to store, including any
  /// required padding. This should not be called, other than to define
  /// byte_size_v, since it is expensive to compile.
  static constexpr auto storage_byte_size() -> std::size_t {
    auto size = byte_sizes[0];
    for (size_t i = 1; i < num_types; ++i) {
      // If not aligned, find the first alignment after the total size:
      if ((size % align_sizes[i]) != 0) {
        auto first_align = align_sizes[i];
        while (first_align < size) {
          first_align += align_sizes[i];
        }
        // New size is smallest alignment.
        size = first_align;
      }
      // Add the size of the component:
      size += byte_sizes[i];
    }
    return size;
  }

  /// Returns the offset, in bytes, to the index I in the type list, accounding
  /// for any required padding dues to __badly specifier__ data layout.
  /// \tparam I The index of the type to get the offset to.
  template <std::size_t I>
  static constexpr auto offset_to() {
    auto offset = std::size_t{0};
    for (std::size_t i = 1; i <= I; ++i) {
      offset += byte_sizes[i - 1];
      // If not aligned, find the first alignment after the total size:
      if ((offset % align_sizes[i]) != 0 || offset < align_sizes[i]) {
        auto first_align = align_sizes[i];
        while (first_align < offset) {
          first_align += align_sizes[i];
        }
        // New size is smallest alignment.
        offset = first_align;
      }
    }
    return offset;
  }

  /// Returns the effective byte size of all elements in the storage.
  static constexpr auto storage_byte_size_v = storage_byte_size();

  //==--- [members] --------------------------------------------------------==//

  buffer_t _data[storage_byte_size_v]; //!< Buffer for the storage.

 public:

  //==--- [interface] ------------------------------------------------------==//

  /// Gets a reference to the Ith data type. This will only be enabled when the
  /// type of the Ith type is not a StorageElement<>.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    non_storage_element_enable_t<T> = 0
  >
  auto get() -> element_value_t<T>& {
    constexpr auto offset = offset_to<I>();
    return *reinterpret_cast<element_value_t<T>*>(
      static_cast<char*>(_data) + offset
    );
  }

  /// Gets a const reference to the Ith data type. This will only be enabled
  /// when the type of the Ith type is not a StorageElement<>.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    non_storage_element_enable_t<T> = 0
  >
  auto get() const -> const element_value_t<T>& {
    constexpr auto offset = offset_to<I>();
    return *reinterpret_cast<const element_value_t<T>*>(
      static_cast<char*>(_data) + offset
    );
  }

  /// Gets a reference to the Jth element of the Ith data type. This will only
  /// be enabled when the type of the Ith type is a StorageElement<> so that the
  /// call to operator[] on the Ith type is valid.
  /// \tparam I The index of the type to get the data from.
  /// \tparam J The index in the type to get.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    std::size_t J,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    storage_element_enable_t<T> = 0
  >
  auto get() -> element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!"
    );
    constexpr auto offset = offset_to<I>();
    return reinterpret_cast<element_value_t<T>*>(
      static_cast<char*>(_data) + offset
    )[J];
  }

  /// Gets a const reference to the Jth element of the Ith data type. This will
  /// only be enabled when the type of the Ith type is a StorageElement<> so
  /// that the call to operator[] on the Ith type is valid.
  /// \tparam I The index of the type to get the data from.
  /// \tparam J The index in the type to get.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    std::size_t J,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    storage_element_enable_t<T> = 0
  >
  auto get() const -> const element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!"
    );
    constexpr auto offset = offset_to<I>();
    return reinterpret_cast<const element_value_t<T>*>(
      static_cast<char*>(_data) + offset
    )[J];
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_OWNED_STORAGE_HPP

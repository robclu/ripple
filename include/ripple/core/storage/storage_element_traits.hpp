//==--- ripple/core/storage/storage_element_traits.hpp ----- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_element_traits.hpp
/// \brief This file traits for elements to be stored.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STORAGE_ELEMENT_TRAITS_HPP
#define RIPPLE_STORAGE_STORAGE_ELEMENT_TRAITS_HPP

#include "storage_layout.hpp"
#include <ripple/core/utility/portability.hpp>
#include <utility>

namespace ripple {

/// The StorageElementTraits struct defines traits for elements which can be
/// stored. This is the default case which is used to define the storage traits
/// for any type that does not have specialized storage.
/// \tparam T The type to specify the storage element traits for.
template <typename T>
struct StorageElementTraits {
  /// Defines the value type of the element, which is the type to store.
  using Value = T;
  /// Defines the number of values of the element to store.
  static constexpr auto num_elements = 1;
  /// Defines the byte size required to allocate the element.
  static constexpr auto byte_size = sizeof(Value);
  /// Defines the alignment size required for the type.
  static constexpr auto align_size = alignof(Value);
  /** Defines that the type is not a storage element. */
  static constexpr bool is_storage_element = false;
};

/// Specialization for a vector implementation.
template <typename T, size_t Values>
struct StorageElementTraits<Vector<T, Values>> {
  /** Defines the value type of the element, which is the type to store. */
  using Value = T;

  // clang-format off
  /** Defines the number of values of the element to store. */
  static constexpr auto num_elements   = Values;
  /** Defines the byte size required to allocate the element. */
  static constexpr auto byte_size      = sizeof(Value) * num_elements;
  /** Defines the alignment requirement for the storage. */
  static constexpr auto align_size     = alignof(Value);
  /** Defines that the type is a vector element. */
  static constexpr bool is_vec_element = true;
  // clang-format on
};

//==--- [is storage element] -----------------------------------------------==//

namespace detail {

/**
 * Defines a class to determine of the type T is a layout type.
 * \tparam T The type to determine if is a storage layout type.
 */
template <typename T>
struct IsStorageLayout : std::false_type {
  /** Defines that the type is not a storage layout type. */
  static constexpr bool value = false;
};

/**
 * Specialization for the case of the IsStorageLayout struct for the case the
 * type to check is a StorageLayout.
 * \tparam Layout The kind of the layout for the storage.
 */
template <LayoutKind Layout>
struct IsStorageLayout<StorageLayout<Layout>> : std::true_type {
  /** Defines that the type is a storage layout type. */
  static constexpr bool value = true;
};

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

} // namespace detail

/**
 * Returns true if the type T is a vector storage element, otherwise returns
 * false.
 * \tparam T The type to determine if is a Vector storage element type.
 */
template <typename T>
static constexpr auto is_vec_element_v =
  StorageElementTraits<std::decay_t<T>>::is_vec_element;

/*==-- [aliases] -----------------------------------------------------------==*/

/**
 * Alias for storage element traits.
 * tparam T The type to get the traits for.
 */
template <typename T>
using storage_element_traits_t = StorageElementTraits<std::decay_t<T>>;

/*==--- [enables] ----------------------------------------------------------==*/

/**
 * Define a valid type if the type T is a vector storage element, otherwise does
 * not define a valid type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using vec_element_enable_t = std::enable_if_t<is_vec_element_v<T>, int>;

/**
 * Define a valid type if the type T is *not* a vector storage element,
 * otherwise does not define a valid type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_vec_element_enable_t = std::enable_if_t<!is_vec_element_v<T>, int>;

namespace detail {

/**
 * Helper class for contiguous storage of multiple type.
 * \tparam Ts the types to store contiguously.
 */
template <typename... Ts>
struct ContigStorageHelper {
  /**
   * Small vector type for constexpr constexts which can be used on both
   * the host and the device to get the offset to the types at compile time.
   * \tparam N The number of elements for the vector.
   */
  template <size_t N>
  struct Vec {
    /**
     * Constructor to set the values of the vector elements.
     * \param  as The values for the vector.
     * \tparam As The type of the elements.
     */
    template <typename... As>
    constexpr Vec(As&&... as) noexcept : data{static_cast<size_t>(as)...} {}

    /**
     * Overload of subscript operator to get the ith value.
     * \param i The index of the element to get.
     * \return The value of the ith element.
     */
    constexpr auto operator[](size_t i) const -> size_t {
      return data[i];
    }

    size_t data[N] = {}; //!< Data for the vector.
  };

  /** Defines the number of different types. */
  static constexpr size_t num_types = sizeof...(Ts);

  /**
   * Gets the number of components for the storage element.
   * \tparam T The type to get the size of.
   */
  template <typename T>
  static constexpr size_t element_components_v =
    storage_element_traits_t<T>::num_elements;

  /** Defines the sizes of each of the types. */
  static constexpr size_t byte_sizes[num_types] = {
    storage_element_traits_t<Ts>::byte_size...};

  /** Defines the alignment sizes of each of the types. */
  static constexpr size_t align_sizes[num_types] = {
    storage_element_traits_t<Ts>::align_size...};

  /** Defines the number of components in each of the types. */
  static constexpr size_t components[num_types] = {element_components_v<Ts>...};

  /**
   * Determines the effective byte size of all elements to store, including any
   * required padding. This should not be called, other than to define
   * byte_size_v, since it is expensive to compile.
   * \return The number of bytes required for the storage.
   */
  static constexpr auto storage_byte_size() noexcept -> size_t {
    auto size = byte_sizes[0];
    for (size_t i = 1; i < num_types; ++i) {
      // If not aligned, find the first alignment after the total size:
      if ((size % align_sizes[i]) != 0) {
        size_t next_align = align_sizes[i];
        while (next_align < size) {
          next_align += align_sizes[i];
        }
        size = next_align;
      }
      size += byte_sizes[i];
    }
    return size;
  }

  /**
   * Determines the offset, in bytes, to the index I in the type list,
   * accounting for any required padding due to *badly specified* data layout.
   * \tparam I The index of the type to get the offset to.
   * \return The offset, in bytes, to the Ith element.
   */
  template <size_t I>
  static constexpr auto offset_to() {
    auto offset = size_t{0};
    for (size_t i = 1; i <= I; ++i) {
      offset += byte_sizes[i - 1];
      // If not aligned, find the first alignment after the total size:
      if ((offset % align_sizes[i]) != 0 || offset < align_sizes[i]) {
        auto next_align = align_sizes[i];
        while (next_align < offset) {
          next_align += align_sizes[i];
        }
        offset = next_align;
      }
    }
    return offset;
  }

  /**
   * Gets an array of offsets for each of the types.
   * \return An array for which each element in the offset in bytes to the Ith
   *         type.
   */
  static constexpr auto offsets() -> Vec<num_types> {
    return make_offsets(std::make_index_sequence<num_types>());
  }

  /**
   * Implementation of offset helper to make the offsets.
   * \tparam Is The indices to use to make the offsets.
   * \return An array for which each element is the offfset in bytes to the Ith
   *         type.
   */
  template <size_t... Is>
  static constexpr auto make_offsets(std::index_sequence<Is...>) {
    return Vec<num_types>{offset_to<Is>()...};
  }
};

} // namespace detail

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_ELEMENT_TRAITS_HPP

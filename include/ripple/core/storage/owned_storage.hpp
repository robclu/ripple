//==--- ripple/core/storage/owned_storage.hpp -------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
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

#include "storage_element_traits.hpp"
#include "storage_traits.hpp"
#include "storage_accessor.hpp"
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {
/**
 * Defines an contiguous, statically allocated, storage which owns the data for
 * the given types.
 * \tparam Ts The types to create storage for.
 */
template <typename... Ts>
class OwnedStorage : public StorageAccessor<OwnedStorage<Ts...>> {
  // clang-format off
  /** Defines the data type for the buffer. */
  using Buffer  = char;
  /** Defines the type of the storage. */
  using Storage = OwnedStorage;
  /** The type of the helper traits class. */
  using Helper  = detail::ContigStorageHelper<Ts...>;
  // clang-format on

  /**
   * Gets the value type of the storage element traits for a type.
   * \tparam T The type to get the storage element traits for.
   */
  template <typename T>
  using element_value_t = typename storage_element_traits_t<T>::Value;

  /*==--- [constants] ------------------------------------------------------==*/

  // clang-format off
  /** Defines the number of different types. */
  static constexpr size_t num_types         = sizeof...(Ts);
  /** Defines the offset to each of the I types.*/
  static constexpr auto   offsets           = Helper::offsets();
  /** Defines the effective byte size of all elements in the storage. */
  static constexpr size_t storage_byte_size = Helper::storage_byte_size();
  /** Alignment of first type. */
  static constexpr size_t first_align       = alignof(nth_element_t<0,Ts...>);
  // clang-format on

  /**
   * Gets the number of components for the storage element.
   * \tparam T The type to get the size of.
   */
  template <typename T>
  static constexpr size_t element_components =
    storage_element_traits_t<T>::num_elements;

  /*==--- [members] --------------------------------------------------------==*/

  /** Buffer for the storage. */
  alignas(first_align) Buffer data_[storage_byte_size] = {};

 public:
  /**
   * Gets the number of components for the nth element.
   * \tparam I The index of the component to get the number of elements for.
   */
  template <size_t I>
  static constexpr size_t nth_element_components =
    element_components<nth_element_t<I, Ts...>>;

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor for the storage.
   */
  constexpr OwnedStorage() noexcept = default;

  /**
   * Constructor to set the owned storage from another type which implements the
   * storage access interface.
   * \param  from The accessor to copy the data from.
   * \tparam Impl The implementation type of the StorageAccessor interface.
   */
  template <typename Impl>
  ripple_host_device constexpr OwnedStorage(
    const StorageAccessor<Impl>& from) noexcept {
    copy(static_cast<const Impl&>(from));
  }

  /*==--- [operator overload] ----------------------------------------------=*/

  /**
   * Overload of operator= to set the data for the owned storage from another
   * type which implements the StorageAccessor interface.
   *
   * \param  from The accessor to copy the data from.
   * \tparam Impl The implementation of the StorageAccessor interface.
   * \return A reference to the OwnedStorage with the copied data.
   */
  template <typename Impl>
  ripple_host_device constexpr auto
  operator=(const StorageAccessor<Impl>& from) noexcept -> OwnedStorage& {
    copy(static_cast<const Impl&>(from));
    return *this;
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Copies the data from the \p other type which must implement the
   * StorageAccessor interface.
   *
   * \param  other The other strided view to copy from.
   * \tparam Other The type of the other storage to copy from.
   */
  template <typename Other>
  ripple_host_device constexpr auto copy(const Other& other) noexcept -> void {
    static_assert(
      is_storage_accessor_v<Other>, "Argument type isn't a StorageAccessor!");
    unrolled_for<num_types>([&](auto i) {
      constexpr size_t type_idx = i;
      using Type                = nth_element_t<type_idx, Ts...>;
      constexpr auto values     = element_components<Type>;
      copy_from_to<type_idx, values, Type>(other, *this);
    });
  }

  /**
   * Determines the number of components in the Ith type being stored. For
   * non-indexable types this will always return 1, otherwise will return the
   * number of possible components which can be indexed.
   *
   * For example:
   *
   * ~~~{.cpp}
   * // Returns 1 -- only 1 type:
   * OwnedStorage<int>().conponents_of<0>();
   *
   * struct A : StridableLayout<A> {
   *  using descriptor_t = StorageElement<int, 4>;
   * };
   * // Returns 4:
   * OwnedStorage<A>().components_of<0>();
   * ~~~
   *
   * \tparam I The index of the type to get the number of components for.
   * \return The number of components in the Ith type.
   */
  template <size_t I>
  ripple_host_device constexpr auto components_of() const noexcept -> size_t {
    return Helper::components[I];
  }

  /**
   * Gets a reference to the Ith data type.
   *
   * \note This is only enabled if the Ith type of the Ith type is not a
   *       StorageElement.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A reference to the Ith type.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Ts...>,
    non_vec_element_enable_t<T> = 0>
  ripple_host_device constexpr auto get() noexcept -> element_value_t<T>& {
    constexpr size_t offset = offsets[I];
    return *static_cast<element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(data_) + offset));
  }

  /**
   * Gets a const reference to the Ith data type.
   *
   * \note This is only enabled if the Ith type of the Ith type is not a
   *       StorageElement.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A reference to the Ith type.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Ts...>,
    non_vec_element_enable_t<T> = 0>
  ripple_host_device constexpr auto
  get() const noexcept -> const element_value_t<T>& {
    constexpr size_t offset = offsets[I];
    return *reinterpret_cast<const element_value_t<T>*>(
      static_cast<const char*>(data_) + offset);
  }

  /**
   * Gets a reference to the Jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \tparam T The type of the Ith element.
   * \return A reference to the Jth element of the Ith type.
   */
  template <
    size_t I,
    size_t J,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device constexpr auto get() noexcept -> element_value_t<T>& {
    static_assert(
      J < element_components<T>, "Out of range acess for storage element!");
    constexpr size_t offset = offsets[I];
    return static_cast<element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(data_) + offset))[J];
  }

  /**
   * Gets a const reference to the Jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \tparam T The type of the Ith element.
   * \return A const reference to the Jth element of the Ith type.
   */
  template <
    size_t I,
    size_t J,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device constexpr auto
  get() const noexcept -> const element_value_t<T>& {
    static_assert(
      J < element_components<T>, "Out of range acess for storage element!");
    constexpr size_t offset = offsets[I];
    return reinterpret_cast<const element_value_t<T>*>(
      static_cast<const char*>(data_) + offset)[J];
  }

  /**
   * Gets a reference to the jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \param  j The index of the component in the type to get.
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A reference to the jth element of the Ith type.
   */
  template <
    size_t I,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device constexpr auto
  get(size_t j) noexcept -> element_value_t<T>& {
    constexpr size_t offset = offsets[I];
    return static_cast<element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(data_) + offset))[j];
  }

  /**
   * Gets a const reference to the jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \param  j The index of the component in the type to get.
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A const reference to the jth element of the Ith type.
   */
  template <
    size_t I,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device constexpr auto
  get(size_t j) const noexcept -> const element_value_t<T>& {
    constexpr size_t offset = offsets[I];
    return reinterpret_cast<const element_value_t<T>*>(
      static_cast<const char*>(data_) + offset)[j];
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_OWNED_STORAGE_HPP

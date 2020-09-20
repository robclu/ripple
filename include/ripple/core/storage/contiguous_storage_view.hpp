//==--- ripple/core/storage/contiguous_storage_view.hpp ---- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  contiguous_storage_view.hpp
/// \brief This file implements a storage class for storing data in a contiguous
///        manner.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_CONTIGUOUS_STORAGE_VIEW_HPP
#define RIPPLE_STORAGE_CONTIGUOUS_STORAGE_VIEW_HPP

#include "owned_storage.hpp"
#include "storage_traits.hpp"
#include "storage_accessor.hpp"

namespace ripple {

/**
 * Implementation of contiguous view storage.
 * \tparam Ts The types to create a storage view for.
 */
template <typename... Ts>
class ContiguousStorageView
: public StorageAccessor<ContiguousStorageView<Ts...>> {
  // clang-format off
  /** Defines the data type for the buffer. */
  using ptr_t           = void*;
  /** Defines the type of the storage. */
  using storage_t       = ContiguousStorageView<Ts...>;
  /** Defines the type of owned storage. */
  using owned_storage_t = OwnedStorage<Ts...>;
  /** The type of the helper traits class. */
  using helper_t        = detail::ContigStorageHelper<Ts...>;
  // clang-format on

  /**
   * Gets the value type of the storage element traits for a type.
   * \tparam T The type to get the storage element traits for.
   */
  template <typename T>
  using element_value_t = typename storage_element_traits_t<T>::value_t;

  /*==--- [constants] ------------------------------------------------------==*/

  // clang-format off
  /** Defines the number of different types. */
  static constexpr auto num_types = sizeof...(Ts);
  /** Defines the offset to each of the I types.*/
  static constexpr auto   offsets             = helper_t::offsets();
  /** Defines the effective byte size of all elements in the storage. */
  static constexpr size_t storage_byte_size_v = helper_t::storage_byte_size();
  // clang-format on

  /**
   * Gets the number of components for the storage element.
   * \tparam T The type to get the size of.
   */
  template <typename T>
  static constexpr auto element_components_v =
    storage_element_traits_t<T>::num_elements;

  /**
   * Gets the number of components for the nth element.
   * \tparam I The index of the component to get the number of elements for.
   */
  template <size_t I>
  static constexpr auto nth_element_components_v =
    element_components_v<nth_element_t<I, Ts...>>;

  /*==--- [allocator] ------------------------------------------------------==*/

  /**
   * Allocator for contiguous storage. This can be used to determine the memory
   * requirement for the storage for different spatial configurations, as well
   * as to offset ContiguousStorage elements within the allocated space.
   */
  struct Allocator {
    /**
     * Determines the number of bytes required to allocate a total of \p
     * elements of the types defined by Ts.
     *
     * \param elements The number of elements to allocate.
     * \return The number of bytes required to allocate the given number of
     *         elements.
     */
    ripple_host_device static constexpr auto
    allocation_size(size_t elements) noexcept -> size_t {
      return storage_byte_size_v * elements;
    }

    /**
     * Determines the number of bytes required to allocate a total of Elements
     * of the types defined by Ts. This overload of the function can be used to
     * allocate static memory when the number of elements in the space is known
     * at compile time.
     * \tparam Elements The number of elements to allocate.
     * \return The number of bytes required to allocate the given number of
     *         elements.
     */
    template <size_t Elements>
    ripple_host_device static constexpr auto
    allocation_size() noexcept -> size_t {
      return storage_byte_size_v * Elements;
    }

    // clang-format off
    /**
     * Offsets the \p storage by the amount specified by the indices \p is. 
     * 
     * \note This assumes that the data into which the storage can offset is
     *       valid, which is the case if the storage was created through the
     *       allocator. If not, the behaviour is undefined.
     * 
     * \param  storage The storage to offset.
     * \param  space   The space for which the storage is defined.
     * \param  is      The indices to offset to in the space.
     * \tparam Space   The implementation of the spatial interface.
     * \tparam Is      The types of the indices.
     * \return A new ContiguousStorageView which points to offset data.
     */
    template <
      typename SpaceImpl, typename... Is, variadic_ge_enable_t<1, Is...> = 0>
    ripple_host_device static auto offset(
      const storage_t&                storage,
      const MultidimSpace<SpaceImpl>& space,
      Is&&...                         is) noexcept -> storage_t {
      storage_t r;
      r._data = static_cast<void*>(
        static_cast<char*>(storage._data) +
        offset_to_aos(space, storage_byte_size_v, std::forward<Is>(is)...));
      return r;
    }
    // clang-format on

    /**
     * Offsets the storage by the amount specified by \p amount in the
     * dimension \p dim.
     *
     * This returns a new ContiguousStorage, offset to the new indices in the
     * space.
     *
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     * \return A new ContiguousStorageView which points to the offset data.
     */
    template <typename SpaceImpl, typename Dim, diff_enable_t<Dim, int> = 0>
    ripple_host_device static auto offset(
      const storage_t&                storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) noexcept -> storage_t {
      storage_t r;
      r._data = static_cast<void*>(
        static_cast<char*>(storage._data) +
        amount * storage_byte_size_v * space.step(dim));
      return r;
    }

    /**
     * Shifts the storage by the amount specified by \p amount in the
     * dimension \p dim.
     *
     * \note This is essentially an in-place offset which modifies the \p
     *       storage to point to the new location.
     *
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     */
    template <typename SpaceImpl, typename Dim>
    ripple_host_device static auto shift(
      storage_t&                      storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) noexcept -> void {
      storage._data = static_cast<void*>(
        static_cast<char*>(storage._data) +
        amount * storage_byte_size_v * space.step(dim));
    }

    // clang-format off
    /**
     * Creates the storage, initializing a ContiguousStorageView instance which
     * has its data pointer pointing to the location defined by the indices \p
     * is, from the initial \p ptr. 
     * 
     * \note The memory space should have a size which is returned by the 
     *       `allocation_size()` method, otherwise this may index into undefined
     *       memory.
     * 
     * \param  ptr       A pointer to the beginning of the memory space.
     * \param  space     The multidimensional space which defines the domain.
     * \param  is        The indices of the element in the space to create.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Is        The type of the indices.
     * \return A new ContiguousStorageView which is offset to the location
     *         specified by the indices.
     */
    template <
      typename SpaceImpl, typename... Is, variadic_ge_enable_t<1, Is...> = 0>
    ripple_host_device static auto
    create(void* ptr, const MultidimSpace<SpaceImpl>& space, Is&&... is)
      noexcept -> storage_t {
      storage_t r;
      r._data = static_cast<void*>(
        static_cast<char*>(ptr) +
        offset_to_aos(space, storage_byte_size_v, std::forward<Is>(is)...));
      return r;
    }
    // clang-format on

    /**
     * Creates the storage, initializing a ContiguousStorageView instance which
     * has its data pointer pointing to the \p ptr.
     *
     * \note The memory space should have a size which is returned by the
     *       `allocation_size()` method, otherwise this may index into undefined
     *       memory.
     *
     * \param  ptr       A pointer to the beginning of the memory space.
     * \param  space     The multidimensional space which defines the domain.
     * \tparam SpaceImpl The implementation of the spatial interface.
     */
    template <typename SpaceImpl>
    ripple_host_device static auto
    create(void* ptr, const MultidimSpace<SpaceImpl>& space) noexcept
      -> storage_t {
      storage_t r;
      r._data = ptr;
      return r;
    }
  };

  /*==--- [members] --------------------------------------------------------==*/

  ptr_t _data = nullptr; //!< Pointer to the data.

 public:
  /*==--- [traits] ---------------------------------------------------------==*/

  /** Defines the type of the allocator for creating StridedStorage. */
  using allocator_t = Allocator;

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor for the contguous storage.
   */
  ripple_host_device ContiguousStorageView() noexcept = default;

  /**
   * Set the contiguous storage from a type which implements the StorageAccess
   * interface.
   *
   * \note This will fail at compile time if Impl doesn't implement the
   *       StorageAccessor interface, \sa StorageAccessor.
   *
   * \param  from The accessor to copy the data from.
   * \tparam Impl The implementation of the StorageAccessor.
   */
  template <typename Impl>
  ripple_host_device ContiguousStorageView(const Impl& from) noexcept {
    copy(from);
  }

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Overload of operator= to set the data for the ContiguousStorageView from
   * a type which implements the StorageAccessor interface.
   *
   * \note This will fail at compile time if Impl doesn't implement the
   *       StorageAccessor interface, \sa StorageAccessor.
   *
   * \param  from The accessor to copy the data from.
   * \tparam Impl The implementation of the StorageAccessor.
   * \return A reference to the newly created storage.
   */
  template <typename Impl>
  ripple_host_device auto operator=(const Impl& from) noexcept -> storage_t& {
    copy(from);
    return *this;
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets a pointer to the data for the storage.
   * \return A pointer to the data for the storage.
   */
  ripple_host_device auto data() noexcept -> void* {
    return _data;
  }

  /**
   * Gets a const pointer to the data.
   * \return A const pointer to the data for the storage.
   */
  ripple_host_device auto data() const noexcept -> const void* {
    return _data;
  }

  /**
   * Copies the data from the \p other type which must implement the
   * StorageAccessor interface.
   *
   * \note The will assert at compile time if \p other does not implement
   *       the StorageAccessor interface.
   *
   * \param  other The other contiguous view to copy from.
   * \tparam Other The type of the other storage to copy from.
   */
  template <typename Other>
  ripple_host_device auto copy(const Other& other) noexcept -> void {
    static_assert(
      is_storage_accessor_v<Other>, "Argument type isn't a StorageAccessor!");

    unrolled_for<num_types>([&](auto i) {
      constexpr std::size_t type_idx = i;
      constexpr auto        values =
        element_components_v<nth_element_t<type_idx, Ts...>>;

      copy_from_to<type_idx, values>(other, *this);
    });
  }

  /**
   * Gets the number of components in the Ith type being stored. For
   * non-indexable types this will always return 1, otherwise it will return the
   * number of possible components which can be indexed.
   *
   * For example:
   *
   * ~~~{.cpp}
   * // Returns 1 -- only 1 type:
   * ContiguousStorageView<int>().conponents_of<0>();
   *
   * struct A : StridableLayout<A> {
   *  using descriptor_t = StorageElement<int, 4>;
   * };
   * // Returns 4:
   * ContiguousStorageView<A>().components_of<0>();
   * ~~~
   *
   * \tparam I The index of the type to get the number of components for.
   * \return The number of components in the type I.
   */
  template <size_t I>
  ripple_host_device constexpr auto components_of() const noexcept -> size_t {
    return helper_t::components[I];
  }

  /**
   * Gets a reference to the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is not a
   *       StorageElement<>.
   *
   * \note All offsetting calculations are performed at compile time.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A reference to the Ith type, if the type is not a StorageElement.
   */
  template <
    size_t I,
    typename T                      = nth_element_t<I, Ts...>,
    non_storage_element_enable_t<T> = 0>
  ripple_host_device auto get() noexcept -> element_value_t<T>& {
    constexpr auto offset = offsets[I];
    return *static_cast<element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(_data) + offset));
  }

  /**
   * Gets a const reference to the Ith data type.
   *
   * \note This will only be enabled  when the type of the Ith type is not a
   *       StorageElement<>.
   *
   * \note All offsetting calculations are performed at compile time.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A const reference to the Ith type, if the type is not a
   *         StorageElement.
   */
  template <
    size_t I,
    typename T                      = nth_element_t<I, Ts...>,
    non_storage_element_enable_t<T> = 0>
  ripple_host_device auto get() const noexcept -> const element_value_t<T>& {
    constexpr auto offset = offsets[I];
    return *static_cast<const element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(_data) + offset));
  }

  /**
   * Gets a reference to the Jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement<> so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \note All offsetting calculations are performed at compile time.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \tparam T The type of the Ith element.
   * \return A reference to the Jth element of the Ith type.
   */
  template <
    size_t I,
    size_t J,
    typename T                  = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0>
  ripple_host_device auto get() noexcept -> element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!");
    constexpr auto offset = offsets[I];
    return static_cast<element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(_data) + offset))[J];
  }

  /**
   * Gets a const reference to the Jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement<> so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \note All offsetting calculations are performed at compile time.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \tparam T The type of the Ith element.
   * \return A const reference to the Jth element of the Ith type.
   */
  template <
    size_t I,
    size_t J,
    typename T                  = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0>
  ripple_host_device auto get() const noexcept -> const element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!");
    constexpr auto offset = offsets[I];
    return static_cast<const element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(_data) + offset))[J];
  }

  /**
   * Gets a reference to the jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement<> so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \note The offset of j is performed at runtime.
   *
   * \param  j The index of the component in the type to get.
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A reference to the jth element of the Ith type.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0>
  ripple_host_device auto get(size_t j) noexcept -> element_value_t<T>& {
    constexpr auto offset = offsets[I];
    return static_cast<element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(_data) + offset))[j];
  }

  /**
   * Gets a const reference to the jth element of the Ith data type.
   *
   * \note This will only be enabled when the type of the Ith type is a
   *       StorageElement<> so that the call to operator[] on the Ith type is
   *       valid.
   *
   * \param  j The index of the component in the type to get.
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A const reference to the jth element of the Ith type.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0>
  ripple_host_device auto
  get(size_t j) const noexcept -> const element_value_t<T>& {
    constexpr auto offset = offsets[I];
    return static_cast<const element_value_t<T>*>(
      static_cast<void*>(static_cast<char*>(_data) + offset))[j];
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_CONTIGUOUS_STORAGE_VIEW_HPP

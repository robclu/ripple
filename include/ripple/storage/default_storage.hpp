/**=--- ripple/storage/default_storage.hpp ----------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  default_storage.hpp
 * \brief This file defines a class for defauly storage.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_STORAGE_DEFAULT_STORAGE_HPP
#define RIPPLE_STORAGE_DEFAULT_STORAGE_HPP

#include <ripple/space/multidim_space.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/**
 * Default storage to store a type T.
 * \tparam T The type for allocation.
 */
template <typename T>
struct DefaultStorage {
  /** The type of the storage. */
  using Storage = T*;

  /**
   * Allocator for default storage. This can be used to determine the memory
   * requirement for the storage for different spatial configurations, as well
   * as to offset DefaultStorage elements within the allocated space.
   */
  struct Allocator {
    /**
     * Gets the number of types which are stored strided.
     * \return The number of types which are stored strided.
     */
    static constexpr auto strided_types() noexcept -> size_t {
      return 1;
    }

    /**
     * Gets the number of elements in the Ith type.
     * \tparam I The index of the type.
     * \return The number of elements in the ith type.
     */
    template <size_t I>
    static constexpr auto num_elements() noexcept -> size_t {
      return 1;
    }

    /**
     * Gets the number of bytes for an elemeent in the Ith type.
     * \tparam I The index of the type to get the element size of.
     * \return The number of bytes for an element in the Ith type.
     */
    template <size_t I>
    static constexpr auto element_byte_size() noexcept -> size_t {
      return sizeof(T);
    }

    /**
     * Gets the number of bytes required to allocate a total of \p elements
     * of the types defined by Ts.
     * \param  elements The number of elements to allocate.
     * \return The number of bytes required to allocate the given number of
     *         elements.
     */
    ripple_host_device static constexpr auto
    allocation_size(size_t elements) noexcept -> size_t {
      return sizeof(T) * elements;
    }

    /**
     * Gets the number of bytes required to allocate a total of Elements
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
      return sizeof(T) * Elements;
    }

    /**
     * Offsets the storage by the amount specified by \p amount in the
     * dimension \p dim.
     *
     *
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     * \return A new DefaultStorage offset by the given amount.
     */
    template <typename SpaceImpl, typename Dim, diff_enable_t<Dim, int> = 0>
    ripple_host_device static auto offset(
      const Storage&                  storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) -> Storage {
      Storage r;
      r = storage + amount * space.step(ripple_forward(dim));
      return r;
    }

    /**
     * Shifts the storage by the amount specified by \p amount in the
     * dimension \p dim.
     *
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     */
    template <typename SpaceImpl, typename Dim>
    ripple_host_device static auto shift(
      Storage&                        storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) -> void {
      storage = storage + (amount * space.step(dim));
    }

    /**
     * Creates the storage, initializing a T instance which points to \p ptr.
     *
     * \param  ptr       A pointer to the beginning of the memory space.
     * \param  space     The multidimensional space which defines the domain.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \return A new DefaultStorage type pointing to the given pointer.
     */
    template <typename SpaceImpl>
    ripple_host_device static auto
    create(void* ptr, const MultidimSpace<SpaceImpl>& space) -> Storage {
      return static_cast<T*>(ptr);
    }
  };

  /**
   * Defines the number of components for the Nth element, which in this case
   * is always 1.
   * \tparam I The index of the element to get the number of components for.
   */
  template <size_t I>
  static constexpr size_t nth_element_components_v = 1;
};

} // namespace ripple

#endif // RIPPLE_STORAGE_DEFAULT_STORAGE_HPP

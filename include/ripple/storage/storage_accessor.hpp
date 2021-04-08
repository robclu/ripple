/**=--- ripple/storage/storage_accessor.hpp ---------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  storage_accessor.hpp
 * \brief This file implements defines an interface for accessing storage
 *        with compile time indices.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_STORAGE_STORAGE_ACCESSOR_HPP
#define RIPPLE_STORAGE_STORAGE_ACCESSOR_HPP

#include "storage_traits.hpp"

namespace ripple {

/**
 * The StorageAccessor struct defines an interface for all classes which
 * access storage using compile time indices, where the compile time indices
 * are used to access the Ith type in the storage, and for storage where the
 * Ith type is a vectortype, the Jth element.
 *
 * \tparam Impl The implementation of the interface.
 */
template <typename Impl>
struct StorageAccessor {
 private:
  /**
   * Gets a constant pointer to the implementation type.
   * \return A const pointer to the implementation type.
   */
  ripple_all constexpr auto impl() const noexcept -> const Impl* {
    return static_cast<const Impl*>(this);
  }

  /**
   * Gets a pointer to the implementation type.
   * \return A pointer to the implementation type.
   */
  ripple_all constexpr auto impl() noexcept -> Impl* {
    return static_cast<Impl*>(this);
  }

 public:
  /**
   * Explicitly copies the data from the other storage type.
   * \param  other     The other storage type tto copy from.
   * \tparam ImplOhter The type of the other storage.
   */
  template <typename Other>
  ripple_all auto copy(const Other& other) noexcept -> void {
    impl()->copy(other);
  }

  /**
   * Returns the number of components in the Ith type being stored. For
   * non-indexable types this will always return 1, otherwise will return the
   * number of possible components which can be indexed.
   * \tparam I The index of the type to get the number of components for.
   * \return The number of components in the Ith element.
   */
  template <size_t I>
  ripple_all constexpr auto components_of() const noexcept -> size_t {
    return impl()->template components_of<I>();
  }

  /**
   * Gets a reference to the Ith data type which is stored.
   * \tparam I The index of the type to get the data from.
   * \return A reference to the Ith type.
   */
  template <size_t I>
  ripple_all decltype(auto) get() noexcept {
    return impl()->template get<I>();
  }

  /**
   * Gets a const reference to the Ith data type which is stored.
   * \tparam I The index of the type to get the data from.
   * \return A const reference to the Ith type.
   */
  template <size_t I>
  ripple_all decltype(auto) get() const noexcept {
    return impl()->template get<I>();
  }

  /**
   * Gets a reference to the Jth element in the Ith data type which is stored.
   *
   * \note The implementation should ensure that calling this with indices I, J
   *       which results in incorrect access causes a compile time error. For
   *       example if the Ith type is an int, for which there is no J element,
   *       then this should cause a compile time error.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index of the element in the type to get.
   * \return A reference to the Jth component of the Ith type.
   */
  template <size_t I, size_t J>
  ripple_all decltype(auto) get() noexcept {
    return impl()->template get<I, J>();
  }

  /**
   * Gets a const reference to the Jth element in the Ith data type which is
   * stored.
   *
   * \note The implementation should ensure that calling this with indices I,
   *       J which results in incorrect access causes a compile time error. For
   *       example if the Ith type is an int, for which there is no J element,
   *       then this should cause a compile time error.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index of the element in the type to get.
   * \return A const reference to the Jth component of the Ith type.
   */
  template <size_t I, size_t J>
  ripple_all decltype(auto) get() const noexcept {
    return impl()->template get<I, J>();
  }

  /**
   * Gets a reference to the jth element in the Ith data type which is stored.
   * \param  j The index of the element in the type to get.
   * \tparam I The index of the type to get the data from.
   * \return A reference to the jth component of the Ith type.
   */
  template <size_t I>
  ripple_all decltype(auto) get(size_t j) noexcept {
    return impl()->get<I>(j);
  }

  /**
   * Gets a reference to the jth element in the Ith data type which is stored.
   * \param  j The index of the element in the type to get.
   * \tparam I The index of the type to get the data from.
   * \return A const reference to the jth component of the Ith type.
   */
  template <size_t I>
  ripple_all decltype(auto) get(size_t j) const noexcept {
    return impl()->get<I>(j);
  }
};

/*==--- [utilities] --------------------------------------------------------==*/

/**
 * Defines a function to copy the Values elements of the Ith type in ImplFrom to
 * elements of the Ith type in ImplTo.
 *
 * \note The calling function should ensure that the Ith type of both ImplTo and
 *       ImplFrom have the same number of elements.
 *
 * \note This overload is only enabled when the Ith type is a vector type and
 *       is indexable.
 *
 * \param  from     The type to copy from.
 * \param  to       The type to copy to.
 * \tparam I        The index of the type for get from \p to and \p from.
 * \tparam Values   The number of values in \p to and \p from for the Ith type.
 * \tparam TypeI    The type of the I element.
 * \tparam ImplFrom The type of the implementation of the \p from type.
 * \tparam ImplTo   The type of the implementation for the \p to type.
 */
template <
  size_t I,
  size_t Values,
  typename TypeI,
  typename ImplFrom,
  typename ImplTo,
  vec_element_enable_t<TypeI> = 0>
ripple_all auto
copy_from_to(const ImplFrom& from, ImplTo& to) noexcept -> void {
  unrolled_for<Values>([&](auto j) {
    constexpr auto J        = size_t{j};
    to.template get<I, J>() = from.template get<I, J>();
  });
}

/**
 * Defines a function to copy the Values elements of the Ith type in ImplFrom to
 * the elements of the Ith type in ImplTo.
 *
 * \note The calling function should ensure that the Ith type of both ImplTo and
 *       ImplFrom have the same number of elements.
 *
 * \note This overload is only enabled when the Ith type is *not* a vector
 *       element and is therefore not indexable.
 *
 * \param  from     The type to copy from.
 * \param  to       The type to copy to.
 * \tparam I        The index of the type for get from \p to and \p from.
 * \tparam Values   The number of values in \p to and \p from for the Ith type.
 * \tparam TypeI    The type of the I element.
 * \tparam ImplFrom The type of the implementation of the \p from type.
 * \tparam ImplTo   The type of the implementation for the \p to type.
 */
template <
  size_t I,
  size_t Values,
  typename TypeI,
  typename ImplFrom,
  typename ImplTo,
  non_vec_element_enable_t<TypeI> = 0>
ripple_all auto copy_from_to(const ImplFrom& from, ImplTo& to) -> void {
  to.template get<I>() = from.template get<I>();
}

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_ACCESSOR_HPP

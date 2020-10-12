//==--- ripple/core/storage/struct_accessor.hpp ------------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  struct_accessor.hpp
/// \brief This file implements defines an interface for accessing a component
///        of a struct at a given index.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STRUCT_ACCESSOR_HPP
#define RIPPLE_STORAGE_STRUCT_ACCESSOR_HPP

#include <ripple/core/utility/portability.hpp>

namespace ripple {

/**
 * The struct accessor gets a components of a struct at a given index for a
 * given storage type, this can be used in a union to allow aliases of data
 * elements, for example:
 *
 * ~~~{.cpp}
 * struct Vec2 {
 *  using Storage = std::array<int, 2>;
 *
 *  union {
 *    Storage data;
 *    StructAccessor<int, Storage, 0> x;
 *    StructAccessor<int, Storage, 1> y;
 *  };
 *
 *  auto operator[](size_t i) -> int& {
 *    return data[i];
 *  }
 * };
 *
 * // Usage:
 * Vec 2 v;
 * v[0] = 2;
 * v.y = 3;
 * ~~~
 * \tparam T       The type of the data for the accessor.
 * \tparam Storage The type of the storage for the accessor.
 * \tparam Index   The index of the element in the storage.
 */
template <typename T, typename Storage, size_t Index>
struct StructAccessor : public Storage {
  /** True if the type is a storage accessor. */
  static constexpr bool is_accessor = is_storage_accessor_v<Storage>;

  /**
   * Overload of conversion operator.
   */
  ripple_host_device operator T() const noexcept {
    if constexpr (is_accessor) {
      return get_accessor<Index>();
    } else {
      return get_nonaccessor<Index>();
    }
  }

  /**
   * Overload of conversion to reference operator.
   */
  ripple_host_device operator T&() noexcept {
    if constexpr (is_accessor) {
      return get_accessor<Index>();
    } else {
      return get_nonaccessor<Index>();
    }
  }

  /**
   * Overload of equal operator to set the value to \p v.
   * \param v The value to set the accessor to.
   */
  ripple_host_device auto operator=(T v) noexcept -> StructAccessor& {
    if constexpr (is_accessor) {
      get_accessor<Index>() = v;
    } else {
      get_nonaccessor<Index>() = v;
    }
    return *this;
  }

 private:
  /**
   * Gets a reference to the data at a given index, if the storage is a storage
   * accessor.
   *
   * \note This is enabled if the storage is a storage accessor.
   *
   * \tparam I The index of the element to get.
   * \return A const reference to the element at the index.
   */
  template <size_t I>
  ripple_host_device auto get_accessor() const noexcept -> const T& {
    static_assert(
      I < Storage::template nth_element_components_v<0>,
      "Invalid index for accessor!");
    return Storage::template get<0, I>();
  }

  /**
   * Gets a reference to the data at a given index, if the storage is a storage
   * accessor.
   *
   * \note This is enabled if the storage is not a storage accessor.
   *
   * \tparam I The index of the element to get.
   * \return A const reference to the element at the index.
   */
  template <size_t I>
  ripple_host_device auto get_nonaccessor() const noexcept -> const T& {
    return Storage::operator[](I);
  }

  /**
   * Gets a reference to the data at a given index, if the storage is a storage
   * accessor.
   *
   * \note This is enabled if the storage is a storage accessor.
   *
   * \tparam I The index of the element to get.
   * \return A const reference to the element at the index.
   */
  template <size_t I>
  ripple_host_device auto get_accessor() noexcept -> T& {
    static_assert(
      I < Storage::template nth_element_components_v<0>,
      "Invalid index for accessor!");
    return Storage::template get<0, I>();
  }

  /**
   * Gets a reference to the data at a given index, if the storage is a storage
   * accessor.
   *
   * \note This is enabled if the storage is not a storage accessor.
   *
   * \tparam I The index of the element to get.
   * \return A const reference to the element at the index.
   */
  template <size_t I>
  ripple_host_device auto get_nonaccessor() noexcept -> T& {
    return Storage::operator[](I);
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_STRUCT_ACCESSOR_HPP

/**=--- ripple/container/vec.hpp --------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  vec.hpp
 * \brief This file defines an implementation for a vector, which implements
 *        the Array interface.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_CONTAINER_VEC_HPP
#define RIPPLE_CONTAINER_VEC_HPP

#include "array.hpp"
#include "array_traits.hpp"
#include "tuple.hpp"
#include <ripple/storage/polymorphic_layout.hpp>
#include <ripple/storage/storage_descriptor.hpp>
#include <ripple/storage/storage_traits.hpp>
#include <ripple/storage/struct_accessor.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/**
 * The VecImpl class implements the Array interface for a vector with a fixed
 * type and known compile time size.
 *
 * The data for the elements is allocated according to the layout, and can be
 * contiguous, owned, or strided.
 *
 * \note This class should not be used directly, use the Vec aliases.
 *
 * \tparam T      The type of the elements in the vector.
 * \tparam Size   The size of the vector.
 * \tparam Layout The type of the storage layout for the vector.
 */
template <typename T, typename Size, typename Layout>
struct VecImpl : public PolymorphicLayout<VecImpl<T, Size, Layout>>,
                 public Array<VecImpl<T, Size, Layout>> {
 private:
  /*==--- [constants] ------------------------------------------------------==*/

  /** Defines the number of elements in the vector. */
  static constexpr auto elements = size_t{Size::value};

  //==--- [aliases] --------------------------------------------------------==//

  // clang-format off
  /** Defines the type of the descriptor for the storage. */
  using Descriptor = StorageDescriptor<Layout, Vector<T, elements>>;
  /** Defines the storage type for the array. */
  using Storage    = typename Descriptor::Storage;
  /** Defines the value type of the data in the vector. */
  using Value      = std::decay_t<T>;

  /** Alias for zeroth accessor */
  using ZerothAccessor = StructAccessor<Value, Storage, 0>;
  /** Alias for first accessor */
  using FirstAccessor  = StructAccessor<Value, Storage, 1>;
  /** Alias for second accessor */
  using SecondAccessor = StructAccessor<Value, Storage, 2>;
  /** Alias for third accessor */
  using ThirdAccessor  = StructAccessor<Value, Storage, 3>;
  // clang-format on

  /**
   * Declares vectors with other storage layouts as friends for construction.
   * \tparam OtherType   The type of the other vector data.
   * \tparam OtherSize   The size of the other vector.
   * \tparam OtherLayout The layout of the other vector.
   */
  template <typename OtherType, typename OtherSize, typename OtherLayout>
  friend struct VecImpl;

  /**
   * LayoutTraits is a friend so that it can see the descriptor.
   * \tparam Layable     If the type can be re laid out.
   * \tparam IsStridable If the type is stridable.
   */
  template <typename Layable, bool IsStridable>
  friend struct LayoutTraits;

 public:
  /*
   * NOTE: Storage accessors are provided for xyzw, and rgba, for the first 4
   *       elements of the vector. There are only valid if the vector has enough
   *       components, and will assert at compile time if an invalid access is
   *       requested (i.e, accessing w in a 3D vector).
   */
  union {
    Storage        storage_; //!< The storage for the vector.
    ZerothAccessor x;        //!< X component of vector.
    FirstAccessor  y;        //!< Y component of vector.
    SecondAccessor z;        //!< Z component of vector.
    ThirdAccessor  w;        //!< W component of vector.
  };

  /*==--- [construction] ---------------------------------------------------==*/

  /** Default constructor for the vector. */
  ripple_host_device constexpr VecImpl() noexcept {}

  /**
   * Sets all elements of the vector to the value \p val.
   * \param val The value to set all elements to.
   */
  ripple_host_device constexpr VecImpl(T val) noexcept {
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() = val; });
  }

  /**
   * Constructor to create the vector from a list of values.
   *
   * \note This overload is only enabled when the number of elements in the
   *       variadic parameter pack matches the number of elements in the vector.
   *
   * \note The types of the values must be convertible to T.
   *
   * \param  values The values to set the elements to.
   * \tparam Values The types of the values for setting.
   */
  template <typename... Values, variadic_size_enable_t<elements, Values...> = 0>
  ripple_host_device constexpr VecImpl(Values&&... values) noexcept {
    const auto v = Tuple<Values...>{values...};
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() = get<i>(v); });
  }

  /**
   * Constructor to set the vector from other \p storage.
   * \param other The other storage to use to set the vector.
   */
  ripple_host_device constexpr VecImpl(Storage storage) noexcept
  : storage_{storage} {}

  /**
   * Copy constructor to set the vector from another vector.
   * \param other The other vector to use to initialize this one.
   */
  ripple_host_device constexpr VecImpl(const VecImpl& other) noexcept
  : storage_{other.storage_} {}

  /**
   * Move constructor to set the vector from another vector.
   * \param other The other vector to use to initialize this one.
   */
  ripple_host_device constexpr VecImpl(VecImpl&& other) noexcept
  : storage_{std::move(other.storage_)} {}

  /**
   * Copy constructor to set the vector from another vector with a different
   * storage layout.
   * \param  other       The other vector to use to initialize this one.
   * \tparam OtherLayout The layout of the other storage.
   */
  template <typename OtherLayout>
  ripple_host_device constexpr VecImpl(
    const VecImpl<T, Size, OtherLayout>& other) noexcept
  : storage_{other.storage_} {}

  /**
   * Move constructor to set the vector from another vector with a different
   * storage layout.
   * \param  other       The other vector to use to initialize this one.
   * \tparam OtherLayout The layout of the other storage.
   */
  template <typename OtherLayout>
  ripple_host_device constexpr VecImpl(VecImpl<T, Size, OtherLayout>&& other)
  : storage_{other.storage_} {}

  /**
   * Constructor to create the vector from an array of the same type and
   * size.
   * \param  arr  The array to use to create this one.
   * \tparam Impl The implementation of the array interface.
   */
  template <typename Impl>
  ripple_host_device constexpr VecImpl(const Array<Impl>& arr) {
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() = arr[i]; });
  }

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Overload of copy assignment overload to copy the elements from another
   * vector to this vector.
   * \param  other The other vector to copy from.
   * \return A references to the modified vector.
   */
  ripple_host_device auto operator=(const VecImpl& other) noexcept -> VecImpl& {
    storage_ = other.storage_;
    return *this;
  }

  /**
   * Overload of move assignment overload to move the elements from another
   * vector to this vector.
   * \param  other The other vector to move.
   * \return A references to the modified vector.
   */
  ripple_host_device auto operator=(VecImpl&& other) noexcept -> VecImpl& {
    storage_ = std::move(other.storage_);
    return *this;
  }

  /**
   * Overload of copy assignment overload to copy the elements from another
   * vector with a different storage layout to this vector.
   * \param  other       The other vector to copy from.
   * \tparam OtherLayout The layout of the other vector.
   * \return A references to the modified vector.
   */
  template <typename OtherLayout>
  ripple_host_device auto
  operator=(const VecImpl<T, Size, OtherLayout>& other) noexcept -> VecImpl& {
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() = other[i]; });
    return *this;
  }

  /**
   * Overload of copy assignment overload to copy the elements from an
   * array into this vector.
   * \param  arr  The array to copy.
   * \tparam Impl The type of the array implementation.
   * \return A references to the modified vector.
   */
  template <typename Impl>
  ripple_host_device auto
  operator=(const Array<Impl>& arr) noexcept -> VecImpl& {
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() = arr[i]; });
    return *this;
  }

  /**
   * Overload of move assignment overload to copy the elements from another
   * vector to this vector.
   * \param  other       The other vector to move.
   * \tparam OtherLayout The layout of the other vector.
   * \return A references to the modified vector.
   */
  template <typename OtherLayout>
  ripple_host_device auto
  operator=(VecImpl<T, Size, OtherLayout>&& other) noexcept -> VecImpl& {
    storage_ = other.storage_;
    return *this;
  }

  /**
   * Gets a constant reference to the element at position i.
   * \param i The index of the element to get.
   * \return A const reference to the element at position i.
   */
  ripple_host_device constexpr auto
  operator[](size_t i) const noexcept -> const Value& {
    return storage_.template get<0>(i);
  }

  /**
   * Gets a reference to the element at position i.
   * \param i The index of the element to get.
   * \return A reference to the element at position i.
   */
  ripple_host_device constexpr auto operator[](size_t i) noexcept -> Value& {
    return storage_.template get<0>(i);
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets a reference to the ith component of the state.
   * \param i The index of the component to get.
   * \return A reference to the component to get.
   */
  template <typename Index>
  ripple_host_device auto component(Index&& i) noexcept -> Value& {
    if constexpr (ripple::is_cx_number_v<Index>) {
      using Idx = std::decay_t<Index>;
      return storage_.template get<0, Idx::value>();
    } else {
      return storage_.template get<0>(i);
    }
  }

  /**
   * Gets a const reference to the ith component of the state.
   * \param i The index of the component to get.
   * \return A const reference to the component to get.
   */
  template <typename Index>
  ripple_host_device auto component(Index&& i) const noexcept -> const Value& {
    if constexpr (ripple::is_cx_number_v<Index>) {
      using Idx = std::decay_t<Index>;
      return storage_.template get<0, Idx::value>();
    } else {
      return storage_.template get<0>(i);
    }
  }

  /**
   * Gets the number of elements in the vector.
   * \return The number of elements in the vector.
   */
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return elements;
  }

  /**
   * Gets the squared length of the vector.
   * \return The squared length of the vector.
   */
  ripple_host_device constexpr auto length_squared() const noexcept -> Value {
    Value result = 0;
    unrolled_for<elements>(
      [&](auto i) { result += component(i) * component(i); });
    return result;
  }

  /**
   * Gets the length of the vector (L2 norm).
   * \return The length of the vector.
   */
  ripple_host_device constexpr auto length() const noexcept -> Value {
    return std::sqrt(length_squared());
  }

  /**
   * Normalizes the vector.
   */
  ripple_host_device constexpr auto normalize() noexcept -> void {
    const auto scale = Value{1} / length();
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() *= scale; });
  }
};

} // namespace ripple

#endif // namespace RIPPLE_CONTAINER_VEC_HPP

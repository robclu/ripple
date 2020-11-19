//==--- ripple/core/container/mat.hpp ---------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  mat.hpp
/// \brief This file defines an implementation for a matrix.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_MAT_HPP
#define RIPPLE_CONTAINER_MAT_HPP

#include "array.hpp"
#include "array_traits.hpp"
#include "tuple.hpp"
#include <ripple/core/storage/polymorphic_layout.hpp>
#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/core/storage/struct_accessor.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/**
 * The MatImpl class implements a matrix class with polymorphic data layout.
 *
 * The data for the elements is allocated according to the layout, and can be
 * contiguous, owned, or strided.
 *
 * \note This class should not be used directly, use the Mat aliases.
 *
 * \tparam T      The type of the elements in the matrix
 * \tparam Rows   The number of rows for the matrix.
 * \tparam Cols   The number of colums for the matrix.
 * \tparam Layout The type of the storage layout for the matrix.
 */
template <typename T, typename Rows, typeanme Cols, typename Layout>
struct MatImpl : public PolymorphicLayout<MatImpl<T, Rows, Cols, Layout>> {
 private:
  /*==--- [constants] ------------------------------------------------------==*/

  /** Defines the number of elements in the matrix. */
  static constexpr auto elements = size_t{Rows::value * Cols::value};

  //==--- [aliases] --------------------------------------------------------==//

  // clang-format off
  /** Defines the type of the descriptor for the storage. */
  using Descriptor = StorageDescriptor<Layout, Vector<T, elements>>;
  /** Defines the storage type for the array. */
  using Storage    = typename Descriptor::Storage;
  /** Defines the value type of the data in the vector. */
  using Value      = std::decay_t<T>;
  // clang-format on

  /**
   * Declares matrixies with other storage layouts as friends for construction.
   * \tparam OType   The type of the other matrix data.
   * \tparam ORows   The number of rows for the other matrix.
   * \tparam OCols   The number of colums for the other matrix.
   * \tparam OLayout The layout of the other vector.
   */
  template <typename OType, typename ORows, typename OCols, typename OLayout>
  friend struct MatImpl;

  /**
   * LayoutTraits is a friend so that it can see the descriptor.
   * \tparam Layable     If the type can be re laid out.
   * \tparam IsStridable If the type is stridable.
   */
  template <typename Layable, bool IsStridable>
  friend struct LayoutTraits;

 public:
  Storage storage_; //!< The storage for the vector.

  /*==--- [construction] ---------------------------------------------------==*/

  /** Default constructor for the matrix. */
  ripple_host_device constexpr MatImpl() noexcept {}

  /**
   * Sets all elements of the matrix to the given value.
   * \param val The value to set all elements to.
   */
  ripple_host_device constexpr MatImpl(T val) noexcept {
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() = val; });
  }

  /**
   * Constructor to create the matrix from a list of values.
   *
   * \note This overload is only enabled when the number of elements in the
   *       variadic parameter pack matches the number of elements in the matrix.
   *
   * \note The types of the values must be convertible to T.
   *
   * \param  values The values to set the elements to.
   * \tparam Values The types of the values for setting.
   */
  template <typename... Values, variadic_size_enable_t<elements, Values...> = 0>
  ripple_host_device constexpr MatImpl(Values&&... values) noexcept {
    const auto v = Tuple<Values...>{values...};
    unrolled_for<elements>(
      [&](auto i) { storage_.template get<0, i>() = get<i>(v); });
  }

  /**
   * Constructor to set the matrix from other \p storage.
   * \param other The other storage to use to set the matrix.
   */
  ripple_host_device constexpr MatImpl(Storage storage) noexcept
  : storage_{storage} {}

  /**
   * Copy constructor to set the matrix from another matrix.
   * \param other The other matrix to use to initialize this one.
   */
  ripple_host_device constexpr MatImpl(const MatImpl& other) noexcept
  : storage_{other.storage_} {}

  /**
   * Move constructor to set the matrix from another matrix.
   * \param other The other matrix to use to initialize this one.
   */
  ripple_host_device constexpr MatImpl(MatImpl&& other) noexcept
  : storage_{ripple_move(other.storage_)} {}

  /**
   * Copy constructor to set the matrix from another matrix with a different
   * storage layout.
   * \param  other       The other matrix to use to initialize this one.
   * \tparam OtherLayout The layout of the other storage.
   */
  template <typename OtherLayout>
  ripple_host_device constexpr MatImpl(
    const MatImpl<T, Rows, Cols, OtherLayout>& other) noexcept
  : storage_{other.storage_} {}

  /**
   * Move constructor to set the matrix from another matrix with a different
   * storage layout.
   * \param  other       The other matrix to use to initialize this one.
   * \tparam OtherLayout The layout of the other storage.
   */
  template <typename OtherLayout>
  ripple_host_device constexpr MatImpl(
    MatImpl<T, Rows, Cols, OtherLayout>&& other)
  : storage_{ripple_move(other.storage_)} {}

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Overload of copy assignment overload to copy the elements from another
   * matrix to this matrix.
   * \param  other The other matrix to copy from.
   * \return A references to the modified matrix.
   */
  ripple_host_device auto operator=(const MatImpl& other) noexcept -> MatImpl& {
    storage_ = other.storage_;
    return *this;
  }

  /**
   * Overload of move assignment overload to move the elements from another
   * matrix to this matrix.
   * \param  other The other matrix to move.
   * \return A reference to the modified vector.
   */
  ripple_host_device auto operator=(MatImpl&& other) noexcept -> MatImpl& {
    storage_ = ripple_move(other.storage_);
    return *this;
  }

  /**
   * Overload of copy assignment overload to copy the elements from another
   * matrix with a different storage layout to this matrix.
   * \param  other       The other matrix to copy from.
   * \tparam OtherLayout The layout of the other matrix.
   * \return A reference to the modified matrix.
   */
  template <typename OtherLayout>
  ripple_host_device auto
  operator=(const MatImpl<T, Rows, Cols, OtherLayout>& other) noexcept
    -> MatImpl& {
    unrolled_for<elements>([&](auto i) {
      storage_.template get<0, i>() = other.storage_.template get<0, i>();
    });
    return *this;
  }

  /**
   * Overload of move assignment overload to copy the elements from another
   * matrix to this matrix.
   * \param  other       The other matrix to move.
   * \tparam OtherLayout The layout of the other matrix.
   * \return A reference to the modified matrix.
   */
  template <typename OtherLayout>
  ripple_host_device auto
  operator=(MatImpl<T, Rows, Cols, OtherLayout>&& other) noexcept -> MatImpl& {
    storage_ = ripple_move(other.storage_);
    return *this;
  }

  /**
   * Overload of call operator to get the element at the given indices.
   * \param row The row of  the element.
   * \param col The column of the element.
   * \return A reference to the element.
   */
  ripple_host_device auto
  operator()(size_t row, size_t col) noexcept -> Value& {
    return storage_[to_index(row, col)];
  }

  /**
   * Overload of call operator to get the element at the given indices.
   * \param row The row of  the element.
   * \param col The column of the element.
   * \return A const reference to the element.
   */
  ripple_host_device auto
  operator()(size_t row, size_t col) const noexcept -> const Value& {
    return storage_[to_index(row, col)];
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets the number of columns for the matrix.
   * \return The number of columns for the matrix.
   */
  ripple_host_device constexpr auto columns() const noexcept -> size_t {
    return Cols::value;
  }

  /**
   * Gets the number of rows for the matrix.
   * \return The number of rows for the matrix.
   */
  ripple_host_device constexpr auto rows() const noexcept -> size_t {
    return Rows::value;
  }

  /**
   * Gets the element at the row and column indices, where the offset to the
   * element is computed at compile time.
   *
   * \tparam Row The index of row for the element.
   * \param  Col The index of the column for the element.
   * \return A const reference to the element at position I.
   */
  template <size_t Row, size_t Col>
  ripple_host_device constexpr auto at() const noexcept -> const Value& {
    static_assert((Row < rows()), "Compile time row index out of range!");
    static_assert((Col < columns()), "Compile time col index out of range!");
    constexpr size_t i = to_element(Row, Col);
    return storage_.template get<0, i>();
  }

  /**
   * Gets the element in the Ith row and Jth column, where the offset to the
   * element is computed at compile time.
   *
   * \tparam Row The index of row for the element.
   * \param  Col The index of the column for the element.
   * \return A reference to the element at position I.
   */
  template <size_t Row, size_t Col>
  ripple_host_device constexpr auto at() const noexcept -> Value& {
    static_assert((Row < rows()), "Compile time row index out of range!");
    static_assert((Col < columns()), "Compile time col index out of range!");
    constexpr size_t i = to_element(Row, Col);
    return storage_.template get<0, i>();
  }

  /**
   * Gets the number of elements in the matrix.
   * \return The number of elements in the matrix.
   */
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return elements;
  }

 private:
  /**
   * Gets the index of the element from the given row and column indices.
   * \param  r The index of the row for the element.
   * \param  c The index of the column for the element.
   * \return The index of the element.
   */
  ripple_host_device constepxr auto
  to_index(size_t r, size_t c) const noexcept -> size_t {
    return r * columns() + c;
  }
};

} // namespace ripple

#endif // namespace RIPPLE_CONTAINER_MATH_HPP
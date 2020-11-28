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

#ifndef RIPPLE_MATH_MAT_HPP
#define RIPPLE_MATH_MAT_HPP

#include "array.hpp"
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
template <typename T, typename Rows, typename Cols, typename Layout>
struct MatImpl;

/**
 * Alias for a matrix with a given number of rows, columns, and layout.
 * \tparam T      The data type of the matrix.
 * \tparam Rows   The number of rows in the matrix.
 * \tparam Cols   The number of columns in the matrix.
 * \tparam Layout The layout of the data for the matrix.
 */
template <
  typename T,
  size_t Rows,
  size_t Cols,
  typename Layout = ContiguousOwned>
using Mat = MatImpl<T, Num<Rows>, Num<Cols>, Layout>;

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
template <typename T, typename Rows, typename Cols, typename Layout>
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
    return storage_.template get<0>(to_index(row, col));
  }

  /**
   * Overload of call operator to get the element at the given indices.
   * \param row The row of  the element.
   * \param col The column of the element.
   * \return A const reference to the element.
   */
  ripple_host_device auto
  operator()(size_t row, size_t col) const noexcept -> const Value& {
    return storage_.template get<0>(to_index(row, col));
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
    constexpr size_t i = to_index(Row, Col);
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
    constexpr size_t i = to_index(Row, Col);
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
  ripple_host_device constexpr auto
  to_index(size_t r, size_t c) const noexcept -> size_t {
    return r * columns() + c;
  }
};

/**
 * Defines the result type for matrix vector multiplication.
 * \tparam Vec The vector type.
 * \tparam Rows The number of rows in the matrix.
 */
template <typename Vec, size_t Rows>
using mat_vec_result_t =
  typename array_traits_t<Vec>::template ImplType<Rows, ContiguousOwned>;

/**
 * Multiplication of a matrix and any array type.
 * \param  m    The matrix to multiply.
 * \param  v    The vector type to multiply with.
 * \tparam T    The type of the data for the matrix.
 * \tparam R    The number of rows in the matrix.
 * \tparam C    The number of columns in the matrix.
 * \tparam L    The layout of the matrix implementation.
 * \tparam Impl The implementation type of the array interface.
 * \return A new vector type which is the result of the multiplication.
 */
template <typename T, typename R, typename C, typename L, typename Impl>
ripple_host_device auto
operator*(const MatImpl<T, R, C, L>& m, const Array<Impl>& v) noexcept
  -> mat_vec_result_t<Impl, R::value> {
  constexpr size_t rows = R::value;
  constexpr size_t cols = C::value;
  using Value           = typename array_traits_t<Impl>::Value;
  using Result          = mat_vec_result_t<Impl, rows>;

  static_assert(
    cols == array_traits_t<Impl>::size,
    "Invalid configuration for matrix vector multiplication!");
  static_assert(
    std::is_convertible_v<T, typename array_traits_t<Impl>::Value>,
    "Matrix and vector types must be convertible!");

  Result result;
  unrolled_for<rows>([&](auto r) {
    result[r] = 0;
    unrolled_for<cols>([&](auto c) { result[r] += m(r, c) * v[c]; });
  });
  return result;
}

/**
 * Multiplication of two matric types.
 * \param  a The left matrix for multiplication.
 * \param  b The right matrix for multiplication.
 * \tparam T1    The type of the data for the first matrix.
 * \tparam T2    The type of the data for the second matrix.
 * \tparam R1    The number of rows in the first matrix.
 * \tparam C1R2  The number of rows/columns in the matrices.
 * \tparam C2    The number of columns in the second  matrix.
 * \tparam L1    The layout of the first matrix implementation.
 * \tparam L2    The layout of the first matrix implementation.
 * \return A new matrix which is the multiplication of the two matrices.
 */
template <
  typename T1,
  typename T2,
  typename R1,
  typename C1R2,
  typename C2,
  typename L1,
  typename L2>
ripple_host_device auto operator*(
  const MatImpl<T1, R1, C1R2, L1>& a,
  const MatImpl<T2, C1R2, C2, L2>& b) noexcept
  -> MatImpl<T1, R1, C2, ContiguousOwned> {
  constexpr size_t rows  = R1::value;
  constexpr size_t cols  = C2::value;
  constexpr size_t inner = C1R2::value;

  static_assert(
    std::is_convertible_v<T1, T2>,
    "Matrix multiplication requires data types which are convertible!");

  using Result = MatImpl<T1, R1, C2, ContiguousOwned>;
  Result res{0};
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      unrolled_for<inner>([&](auto i) { res(r, c) += a(r, i) * b(i, c); });
    }
  }
  return res;
}

} // namespace ripple

#endif // namespace RIPPLE_MATH_MAT_HPP
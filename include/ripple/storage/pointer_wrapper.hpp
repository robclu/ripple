/**=--- ripple/storage/pointer_wrapper.hpp ----------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  pointer_wrapper.hpp
 * \brief This file defines a wrapper class which can be used to provide
 *        pointer like access to a type.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_STORAGE_POINTER_WRAPPER_HPP
#define RIPPLE_STORAGE_POINTER_WRAPPER_HPP

#include <ripple/utility/portability.hpp>

namespace ripple {

/**
 * The PointerWrapper class stores a type T and provides pointer like access to
 * the type T. That is, the semantics of the PointerWrapper and a pointer are
 * the same.
 * \tparam T The type to wrap with pointer like access.
 */
template <typename T>
struct PointerWrapper {
  /**
   * Constructor to create the wrapper.
   * \param data The data to wrap as a pointer.
   */
  ripple_host_device PointerWrapper(T data) noexcept : data_{data} {}

  // clang-format off
  /**
   * Overload of the access operator to access the type T with pointer like
   * syntax.
   * \return A pointer to the wrapped type.
   */
  ripple_host_device auto operator->() noexcept -> T* {
    return &data_;
  }

  /**
   * Overload of the access operator to access the type T with const pointer
   * like access.
   * \return A const pointer to the wrapped type.
   */
  ripple_host_device auto operator->() const noexcept -> const T* {
    return &data_;
  }

 private:
  T data_; //!< The data to wrap as a pointer.
};

} // namespace ripple

#endif // RIPPLE_STORAGE_POINTER_WRAPPER_HPP

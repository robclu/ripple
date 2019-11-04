//==--- ripple/storage/storage_accessor.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_accessor.hpp
/// \brief This file implements defines an interface for accessing storage 
///        with compile time indices.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STORAGE_ACCESSOR_HPP
#define RIPPLE_STORAGE_STORAGE_ACCESSOR_HPP

#include "storage_traits.hpp"

namespace ripple {

/// The StorageAccessor struct defines an interface for all classes which
/// access storage using compile time indices, where the compile time indices
/// are used to access the Ith type in the storage, and for storage where the
/// Ith type is an array type, the Jth element.
/// \tparam Impl The implementation of the interface.
template <typename Impl>
struct StorageAccessor {
 private:
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a constant pointer to the implementation type.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation type.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
  /// Returns the number of components in the Ith type being stored. For
  /// non-indexable types this will always return 1, otherwise will return the
  /// number of possible components which can be indexed.
  /// \tparam I The index of the type to get the number of components for.
  template <std::size_t I>
  ripple_host_device constexpr auto components_of() const -> std::size_t {
    return impl()->template components_of<I>();
  }

  /// Gets a reference to the Ith data type which is stored.
  /// \tparam I The index of the type to get the data from.
  template <std::size_t I>
  ripple_host_device auto get() {
    return impl()->template get<I>();
  }

  /// Gets a const reference to the Ith data type which is stored.
  /// \tparam I The index of the type to get the data from.
  template <std::size_t I>
  ripple_host_device auto get() const {
    return impl()->template get<I>();
  }

  /// Gets a reference to the Jth element in the Ith data type which is stored.
  /// The implementation should ensure that calling this with indices I, J which
  /// result in incorrect acceess, causes a compile time error. For example if
  /// the Ith type is an int, for which there is no J element, then this should
  /// cause a compile time error.
  /// \tparam I The index of the type to get the data from.
  /// \tparam J The index of the element in the type to get.
  template <std::size_t I, std::size_t J>
  ripple_host_device auto get() {
    return impl()->template get<I, J>();
  }

  /// Gets a const reference to the Jth element in the Ith data type which is
  /// stored. The implementation should ensure that calling this with indices I,
  /// J which result in incorrect acceess, causes a compile time error. For
  /// example if the Ith type is an int, for which there is no J element, then
  /// this should cause a compile time error.
  /// \tparam I The index of the type to get the data from.
  /// \tparam J The index of the element in the type to get.
  template <std::size_t I, std::size_t J>
  ripple_host_device auto get() const {
    return impl()->template get<I, J>();
  }

  /// Gets a reference to the jth element in the Ith data type which is stored.
  /// The implementation should ensure that calling this with indices I, j which
  /// result in incorrect acceess, causes a compile time error. For example if
  /// the Ith type is an int, for which there is no j element, then this should
  /// cause a compile time error.
  /// \param  j The index of the element in the type to get.
  /// \tparam I The index of the type to get the data from.
  template <std::size_t I>
  ripple_host_device auto get(std::size_t j) {
    return impl()->get<I>(j);
  }

  /// Gets a reference to the jth element in the Ith data type which is stored.
  /// The implementation should ensure that calling this with indices I, j which
  /// result in incorrect acceess, causes a compile time error. For example if
  /// the Ith type is an int, for which there is no j element, then this should
  /// cause a compile time error.
  /// \param  j The index of the element in the type to get.
  /// \tparam I The index of the type to get the data from.
  template <std::size_t I>
  ripple_host_device auto get(std::size_t j) const {
    return impl()->get<I>(j);
  }
};

//==--- [utilities] --------------------------------------------------------==//

/// Defines a valid type if V > 1, otherwise does not.
/// \tparam V The size to base the enable on.
template <std::size_t V>
using indexable_enable_t = std::enable_if_t<(V > 1), int>;

/// Defines a valid type if V <= 1, otherwise does not.
/// \tparam V The size to base the enable on.
template <std::size_t V>
using non_indexable_enable_t = std::enable_if_t<(V <= 1), int>;

/// Defines a function to copy the Values elements of the Ith type in ImplTo to
/// the Values elements of the Ith type in ImplFrom.
///
/// The calling function should ensure that the Ith type of both ImplTo and
/// ImplFrom are the same, and therefore that Values is the same for both.
///
/// This overload is only enabled when the Ith type in indexable.
///
/// \param  to       The type to copy to.
/// \param  from     The type to copy from.
/// \tparam I        The index of the type for get from \p to and \p from.
/// \tparam Values   The number of values in \p to and \p from for the Ith type.
/// \tparam ImplFrom The type of the implementation of the \p from type.
/// \tparam ImplTo   The type of the implementation for the \p to type.
template <
  std::size_t I        ,
  std::size_t Values   ,
  typename    ImplFrom ,
  typename    ImplTo   ,
  indexable_enable_t<Values> = 0
>
ripple_host_device auto copy_from_to(const ImplFrom& from, ImplTo& to) -> void {
  unrolled_for<Values>([&from, &to] (auto J) {
    to.template get<I, J>() = from.template get<I, J>();
  });
}

/// Defines a function to copy the Values elements of the Ith type in ImplTo to
/// the Values elements of the Ith type in ImplFrom.
///
/// The calling function should ensure that the Ith type of both ImplTo and
/// ImplFrom are the same, and therefore that Values is the same for both.
///
/// This overload is only enabled when the Ith type is not indexable.
///
/// \param  from     The type to copy from.
/// \param  to       The type to copy to.
/// \tparam I        The index of the type for get from \p to and \p from.
/// \tparam Values   The number of values in \p to and \p from for the Ith type.
/// \tparam ImplFrom The type of the implementation of the \p from type.
/// \tparam ImplTo   The type of the implementation for the \p to type.
template <
  std::size_t I       ,
  std::size_t Values  ,
  typename    ImplFrom,
  typename    ImplTo  ,
  non_indexable_enable_t<Values> = 0
>
ripple_host_device auto copy_from_to(const ImplFrom& from, ImplTo& to) -> void {
  to.template get<I>() = from.template get<I>();
}

} // namespace ripple

#endif // RIPPLE_STORAGE_STORAGE_ACCESSOR_HPP

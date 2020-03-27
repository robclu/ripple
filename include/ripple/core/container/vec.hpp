//==--- ripple/core/container/vec.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vec.hpp
/// \brief This file defines an implementation for a vector, which implements
///        the Array interface.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_VEC_HPP
#define RIPPLE_CONTAINER_VEC_HPP

#include "array.hpp"
#include "array_traits.hpp"
#include "tuple.hpp"
#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/storage_element_traits.hpp>
#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/// The Vec class implements the Array interface for a vector with a fixed type
/// and known compile time size. The data for the elements in allocated
/// statically on the heap and is stored contiguously. This vector class should
/// be used on the CPU, or when data is accessed from a register on the GPU.
///
/// \tparam T      The type of the elements in the vector.
/// \tparam Size   The size of the vetor.
/// \tparam Layout The type of the storage layout for the vector.
template <typename T, typename Size, typename Layout>
struct VecImpl : 
  public StridableLayout<VecImpl<T, Size, Layout>>,
  public Array<VecImpl<T, Size, Layout>>  {
 private:
  //==--- [constants] ------------------------------------------------------==//

  /// Defines the number of elements in the vector.
  static constexpr auto elements = size_t{Size::value};

  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of the descriptor for the storage.
  using descriptor_t = StorageDescriptor<Layout, StorageElement<T, elements>>;
  /// Defines the storage type for the array.
  using storage_t    = typename descriptor_t::storage_t;
  /// Defines the value type of the array data.
  using value_t      = std::decay_t<T>;
  /// Defines the type of this vector.
  using self_t       = VecImpl<T, Size, Layout>;

  /// Declares vectors with other storage layouts as friends for construction.
  template <typename OtherType, typename OtherSize, typename OtherLayout>
  friend struct VecImpl;

  /// LayoutTraits is a friend so that it can see the descriptor.
  template <typename Layable, bool IsStridable>
  friend struct LayoutTraits;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor for the vector.
  ripple_host_device constexpr VecImpl() {};
  
  /// Constructor to create the array from a list of values. This overload is
  /// only enabled when the number of elements in the variadic parameter pack
  /// matches the size of the array.
  template <typename... Values, variadic_size_enable_t<elements, Values...> = 0>
  ripple_host_device constexpr VecImpl(Values&&... values) {
    const auto v = Tuple<Values...>{values...};
    unrolled_for<elements>([&] (auto _i) {
      constexpr auto i = size_t{_i};
      _storage.template get<0, i>() = get<i>(v);
    });
  }

  /// Constructor to set the vector from other \p storage.
  /// \param other The other storage to use to set the vector.
  ripple_host_device constexpr VecImpl(storage_t storage) : _storage{storage} {}

  /// Coyp constructor to set the vector from another vector with a potentially
  /// different storage layout.
  /// \tparam OtherLayout The layout of the other storage.
  ripple_host_device constexpr VecImpl(const VecImpl& other)
  : _storage{other._storage} {}

  /// Move constructor to set the vector from another vector with a potentially
  /// different storage layout.
  /// \tparam OtherLayout The layout of the other storage.
  ripple_host_device constexpr VecImpl(VecImpl&& other)
  : _storage{std::move(other._storage)} {}

  /// Coyp constructor to set the vector from another vector with a potentially
  /// different storage layout.
  /// \tparam OtherLayout The layout of the other storage.
  template <typename OtherLayout>
  ripple_host_device constexpr VecImpl(
    const VecImpl<T, Size, OtherLayout>& other
  ) : _storage{other._storage} {}

  /// Move constructor to set the vector from another vector with a potentially
  /// different storage layout.
  /// \tparam OtherLayout The layout of the other storage.
  template <typename OtherLayout>
  ripple_host_device constexpr VecImpl(VecImpl<T, Size, OtherLayout>&& other)
  : _storage{other._storage} {}

  /// Constructor to create the vector from another array of the same type and
  /// size.
  /// \param  arr  The array to use to create this one.
  /// \tparam Impl The implementation of the arrau interface.  
  template <typename Impl>
  ripple_host_device constexpr VecImpl(const Array<Impl>& arr) {
    unrolled_for<elements>([&] (auto _i) {
      constexpr auto i = size_t{_i};
      _storage.template get<0, i>() = arr[i];
    });
  }

  //==--- [operator overloads] ---------------------------------------------==//
  
  /// Overload of copy assignment overload to copy the elements from another
  /// vector with potentially different storage layout to this vector.
  /// \param  other The other vector to copy from.
  ripple_host_device auto operator=(const VecImpl& other) -> self_t& {
    _storage = other._storage;
    return *this;
  }

  /// Overload of move assignment overload to copy the elements from another
  /// vector with potentially different storage layout to this vector.
  /// \param  other The other vector to move.
  ripple_host_device auto operator=(VecImpl&& other) -> self_t& {
    _storage = std::move(other._storage);
    return *this;
  } 
  
  /// Overload of copy assignment overload to copy the elements from another
  /// vector with potentially different storage layout to this vector.
  /// \param  other       The other vector to copy from.
  /// \tparam OtherLayout The layout of the other vector.
  template <typename OtherLayout>
  ripple_host_device auto operator=(const VecImpl<T, Size, OtherLayout>& other)
  -> self_t& {
    unrolled_for<elements>([&] (auto _i) {
      constexpr auto i = size_t{_i};
      _storage.template get<0, i>() = other[i];
    });
    return *this;
  }

  /// Overload of move assignment overload to copy the elements from another
  /// vector with potentially different storage layout to this vector.
  /// \param  other       The other vector to move.
  /// \tparam OtherLayout The layout of the other vector.
  template <typename OtherLayout>
  ripple_host_device auto operator=(VecImpl<T, Size, OtherLayout>&& other)
  -> self_t& {
    _storage = other._storage;
    return *this;
  } 
 
  /// Returns a constant reference to the element at position i.
  /// \param i The index of the element to get.  
  ripple_host_device constexpr auto operator[](size_t i) const 
  -> const value_t& {
    return _storage.template get<0>(i);
  }

  /// Returns a reference to the element at position i.
  /// \param i The index of the element to get.  
  ripple_host_device constexpr auto operator[](size_t i) -> value_t& {
    return _storage.template get<0>(i);
  }

  //==--- [interface] ------------------------------------------------------==//
  
  /// Gets the element at index I, where the offset to the element is computed
  /// at compile time.
  /// \tparam I The index of the element to get.
  template <size_t I>
  ripple_host_device constexpr auto at() const -> const value_t& {
    static_assert((I < elements), "Compile time index out of range!");
    return _storage.template get<0, I>();
  }

  /// Gets the element at index I, where the offset to the element is computed
  /// at compile time.
  /// \tparam I The index of the element to get.
  template <size_t I>
  ripple_host_device constexpr auto at() -> value_t& {
    static_assert((I < elements), "Compile time index out of range!");
    return _storage.template get<0, I>();
  }
   
  /// Returns the number of elements in the vector. 
  ripple_host_device constexpr auto size() const -> size_t {
    return elements;
  }
    
 private:
  storage_t _storage;  //!< The storage for the vector.
};

} // namespace ripple

#endif // namespace RIPPLE_CONTAINER_VEC_HPP



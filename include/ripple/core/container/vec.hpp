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
#include <ripple/core/utility/portability.hpp>

namespace ripple {

/// The Vec class implements the Array interface for a vector with a fixed type
/// and known compile time size. The data for the elements in allocated
/// statically on the heap and is stored contiguously. This vector class should
/// be used on the CPU, or when data is accessed from a register on the GPU.
///
/// \tparam T    The type of the elements in the vector.
/// \tparam Size The size of the vetor.
template <typename T, std::size_t Size>
struct Vec : public Array<Vec<T, Size>>  {
 private:
  /// Defines the type of the traits for the vector.
  using traits_t  = ArrayTraits<Vec<T, Size>>;
  /// Defines the storage type for the array.
  using storage_t = typename traits_t::storage_t;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor for the vector.
  ripple_host_device constexpr Vec() {};
  
  /// Constructor to create the array from a list of values. This overload is
  /// only enabled when the number of elements in the variadic parameter pack
  /// matches the size of the array.
  template <typename... Values, variadic_size_enable_t<Size, Values...> = 0>
  ripple_host_device constexpr Vec(Values&&... values)
  : _data{static_cast<typename traits_t::value_t>(values)...} {}

  /// Constructor to create the vector from another array of the same type and
  /// size.
  /// \param  arr  The array to use to create this one.
  /// \tparam Impl The implementation of the arrau interface.  
  template <typename Impl>
  ripple_host_device constexpr Vec(const Array<Impl>& arr) {
    for (auto i = 0; i < Size; ++i) {
      _data[i] = arr[i];
    }
  }

  //==--- [operator overloads] ---------------------------------------------==//
 
  /// Returns a constant reference to the element at position i.
  /// \param i The index of the element to get.  
  ripple_host_device constexpr auto operator[](std::size_t i) const
  -> const typename traits_t::value_t& {
    return _data[i];
  }

  /// Returns a reference to the element at position i.
  /// \param i The index of the element to get.  
  ripple_host_device constexpr auto operator[](std::size_t i)
  -> typename traits_t::value_t& {
    return _data[i];
  }
   
  /// Returns the number of elements in the vector. 
  ripple_host_device constexpr auto size() const -> std::size_t {
    return Size;
  }
    
 private:
  storage_t _data[Size];  //!< The data for the array.
};

} // namespace ripple

#endif // namespace RIPPLE_CONTAINER_VEC_HPP



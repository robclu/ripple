//==--- streamline/container/vec.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Streamline.
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

#ifndef STREAMLINE_CONTAINER_VEC_HPP
#define STREAMLINE_CONTAINER_VEC_HPP

#include "array.hpp"
#include <streamline/utility/portability.hpp>

namespace streamline {

/// The Vec class implements the Array interface for a vector with a fixed type
/// and known compile time size. The data for the elements in allocated
/// statically on the heap and is contiguous.
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
  /// Default constructor for the vector.
  constexpr Vec() = default;
  
  /// Constructor to create the vector from another array of the same type and
  /// size.
  /// \param[in] arr  The array to use to create this one.
  /// \tparam    Impl The implementation of the arrau interface.  
  template <typename Impl>
  constexpr Vec(const Array<Impl>& arr) {
    for (auto i = 0; i < Size; ++i)
      _data[i] = arr[i];
  }
 
  /// Returns a constant reference to the element at position i.
  /// \param[in] i The index of the element to get.  
  streamline_host_device constexpr auto operator[](std::size_t i) const
  -> const typename traits_t::value_t& {
    return _data[i];
  }

  /// Returns a reference to the element at position i.
  /// \param[in] i The index of the element to get.  
  streamline_host_device constexpr auto operator[](std::size_t i)
  -> typename traits_t::value_t& {
    return _data[i];
  }
    
  streamline_host_device constexpr auto size() const -> std::size_t {
    return Size;
  }
    
 private:
  storage_t _data[Size];  //!< The data for the array.
};

} // namespace streamline

#endif // namespace STREAMLINE_CONTAINER_VEC_HPP



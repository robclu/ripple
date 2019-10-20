//==--- streamline/container/array_traits.hpp -------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Streamline.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array_traits.hpp
/// \brief This file defines forward declarations and traits for arrays.
//
//==------------------------------------------------------------------------==//

#ifndef STREAMLINE_CONTAINER_ARRAY_TRAITS_HPP
#define STREAMLINE_CONTAINER_ARRAY_TRAITS_HPP

#include <streamline/utility/type_traits.hpp>

namespace streamline {

//==--- [Forward declarations] ---------------------------------------------==//

/// The Array class defines an interface to which all specialized
/// implementations must conform. The implementation is provided by
/// the template type Impl.
/// \tparam Impl The implementation of the array interface.
template <typename Impl> struct Array;

/// The Vec class implements the Array interface to store data in AoS format.
/// \tparam T     The type of the data for the vector.
/// \tparam Size  The size of the vector.
template <typename T, std::size_t S> struct Vec;

/// The SoaVec class implements the Array interface to store data in AoS format.
/// \tparam T     The type of the data for the vector.
/// \tparam Size  The size of the vector.
template <typename T, std::size_t S> struct SoaVec;

//==--- [Definitions] ------------------------------------------------------==//

/// Returns true if the template parameter is an Array.
/// \tparam T The type to determine if is an array.
template <typename T>
static constexpr bool is_array_v = std::is_base_of_v<Array<T>, T>;

//==--- [Traits] -----------------------------------------------------------==//

/// The ArrayTraits structs defines traits that all classes which implement the
/// Array interface must implement. This is the default implementation, and
/// defines traits for a type T which does not implement the Array interface.
template <typename T>
struct ArrayTraits {
  /// The underlying type for the array.
  using raw_t     = std::decay_t<T>;
  /// The value type for the array.
  using value_t   = raw_t;
  /// The type used to store data for the array.
  using storage_t = value_t*;
  
  /// Returns the number of elements in the array.  
  static constexpr auto size            = 1;
  /// Returns that the array data cannot be stored as SOA.
  static constexpr auto is_soa          = false;
  /// Returns the number of bytes required to allocate an element.  
  static constexpr auto elem_alloc_size = sizeof(value_t);
};

/// Specialization of the ArrayTraits class for the Vec class.
/// \tparam T    The type of the data for the vec.
/// \tparam Size The size  of the vector.
template <typename T, std::size_t Size>
struct ArrayTraits<Vec<T, Size>> {
  /// The underlying type stored in the vector.
  using raw_t     = std::decay_t<T>;
  /// The value type of the vector.
  using value_t   = Vec<T, Size>;
  /// The type to use to store the vector data.
  using storage_t = raw_t;

  /// Returns the number of elements in the array.  
  static constexpr auto size            = Size;
  /// Returns that the array data cannot be stored as SOA.
  static constexpr auto is_soa          = false;
  /// Returns the number of bytes required to allocate a Vec element.  
  static constexpr auto elem_alloc_size = sizeof(T) * size;
};

/// Specialization of the ArrayTraits class for the SoaVec class.
/// \tparam T    The type of the data for the vec.
/// \tparam Size The size  of the vector.
template <typename T, std::size_t Size>
struct ArrayTraits<SoaVec<T, Size>> {
  /// The underlying type stored in the vector.
  using raw_t     = std::decay_t<T>;
  /// The value type of the vector.
  using value_t   = SoaVec<T, Size>;
  /// The type to use to store the vector data.
  using storage_t = value_t*;

  /// Returns the number of elements in the array.  
  static constexpr auto size            = Size;
  /// Returns that the array data cannot be stored as SOA.
  static constexpr auto is_soa          = true;
  /// Returns the number of bytes required to allocate a Soa element.
  static constexpr auto elem_alloc_size = sizeof(T) * size;
};

/// Specialization of the ArrayTraits class for the case that it's celled using
/// the interface, in which case the implementation is used to redefine the
/// traits.
/// \tparam Impl The implementation to get the traits
template <typename Impl>
struct ArrayTraits<Array<Impl>> {
private:
  /// Defines the type of the implementation traits.
  using impl_traits_t = ArrayTraits<Impl>;
public:
  /// Defines the type stored in the arrap.
  using raw_t     = typename impl_traits_t::raw_t;
  /// Defines the value type of the array.
  using value_t   = typename impl_traits_t::value_t;
  /// Defines the storage type for the array.
  using storage_t = typename impl_traits_t::storage_t;
    
  /// Returns the number of elements in the array.  
  static constexpr auto size            = impl_traits_t::size;
  /// Returns true if the array is an SOA version.
  static constexpr auto is_soa          = impl_traits_t::is_soa;
  /// Returns the number of bytes required to allocate an element.
  static constexpr auto elem_alloc_size = impl_traits_t::elem_alloc_size;
};

/// Gets the array traits for the type T.
/// \tparam T The type to get the array traits for.
template <typename T>
using array_traits_t = ArrayTraits<std::decay_t<T>>;

} // namespace streamline

#endif // STREAMLINE_CONTAINER_ARRAY_TRAITS_HPP

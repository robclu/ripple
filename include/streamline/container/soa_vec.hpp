//==--- streamline/container/soa_vec.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Streamline
// 
//                      Copyright (c) 2019 Streamline.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  soa_vec.hpp
/// \brief This file defines an implementation for a vector, which implements
///        the Array interface, and which stores its members in SOA format.
//
//==------------------------------------------------------------------------==//

#ifndef STREAMLINE_CONTAINER_SOA_VEC_HPP
#define STREAMLINE_CONTAINER_SOA_VEC_HPP

#include "array.hpp"
#include <streamline/utility/portability.hpp>

namespace streamline {

/// The SoaVec class implements the Array interface for a vector with a fixed
/// type and known compile time size. The data for the elements is __not__
/// allocated by the class itself, and can only be set from a pointer to data of
/// the appropriate type, which has already been allocated for an Soa layout.
/// 
/// This class is therefore essesntially a wrapper class which can be used to
/// access the already allocated Soa data as if it was stored as Aos.
///
/// This vector class should be used on the CPU when it's known that SIMD 
/// operations can be performed on the data, and when accessing the data on the
/// GPU from either global or shared memory, but __not__ when the data is
/// accessed from a register.
///
/// \tparam T    The type of the elements in the vector.
/// \tparam Size The size of the vetor.
template <typename T, std::size_t Size>
struct SoaVec : public Array<SoaVec<T, Size>> {
 private:
  /// Defines the type of the traits for the vector.
  using traits_t  = ArrayTraits<Vec<T, Size>>;
  /// Defines the storage type for the array.
  using storage_t = typename traits_t::storage_t;

 public:
  /// Deleted default constructor. SoaVec cannot be created by default, since it
  /// must be initialized with a valid pointer and step size.
  SoaVec() = delete;

  /// Constructor, the SoaVec must be constructed from a pointer with a step
  /// size. This does not check the validity of the pointer, unless in debug
  /// mode. 
  /// \param data A pointer to the storage for the vector.
  /// \param step The size of the step to the next element.
  steamline_host_device explicit SoaVec(storage_t data, std::size_t step)
  : _data{data}, _step{step} {}
 
  /// Constructor to create the vector from another array of the same type and
  /// size.
  /// \param  arr  The array to use to create this one.
  /// \tparam Impl The implementation of the arrau interface.  
  template <typename Impl>
  streamline_host_device constexpr Vec(const Array<Impl>& arr) {
    unrolled_for_bounded<Size>([&] (auto i) {
      _data[i] = arr[i];
    });
  }

  /// Returns a constant reference to the element at position i.
  /// \param i The index of the element to get.  
  streamline_host_device constexpr auto operator[](std::size_t i) const
  -> const typename traits_t::value_t& {
    return _data[i];
  }

  /// Returns a reference to the element at position i.
  /// \param i The index of the element to get.  
  streamline_host_device constexpr auto operator[](std::size_t i)
  -> typename traits_t::value_t& {
    return _data[i];
  }
   
  /// Returns the number of elements in the vector. 
  streamline_host_device constexpr auto size() const -> std::size_t {
    return Size;
  }

 private:
  storage_t   _data = nullptr;  //!< Storage for the vector.
  std::size_t _step = 0;        //!< Step size to the next element.
};

} // namespace streamline

#endif // STREAMLINE_CONTAINER_SOA_VEC_HPP


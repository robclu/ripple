//==--- ripple/container/tensor_storage.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor_storage.hpp
/// \brief This file imlements tensor storage functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_TENSOR_STORAGE_HPP
#define RIPPLE_CONTAINER_TENSOR_STORAGE_HPP

#include "array_traits.hpp"
#include "tensor_traits.hpp"

namespace ripple {

/// A storage class for tensors of a specific type and dimensionaliy. This class
/// holds the storage and size of the dimensions for a tensor, but does not
/// allocate the memory for the storage. The allocation of the storage must be
/// performed by the class using the storage.
///
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions>
class TensorStorage {
 private:
  /// Defines the type to store in the tensor. For types which are not Array
  /// implementations, this will just be the type, however, it's possible that
  /// this may be some underlying type, such as when the data can be stored as
  /// SOA.
  using storage_t   = typename array_traits_t<T>::storage_t;
  /// Defines the sizes of the dimensions for the tensor.
  using dim_sizes_t = Vec<std::size_t, Dimensions>;

 public:
  /// Default constructor, which can be used in the case that the sizes of the
  /// dimensions of the tensor are not known, and can then be resized later.
  TensorStorage() = default;

  /// Initializes the size of each of the dimensions of the tensor. This is only
  /// enabled when the number of arguments matches the dimensionality of the
  /// tensor storage, and the sizes are numeric types.
  ///
  /// This does not allocate the data for the storage, that needs to be done by
  /// the class which uses the storage.
  ///
  /// \param size_0 The size of the zero dimension for the tensor.
  /// \param sizes  The sizes of the other dimensions of the tensor, if there
  ///               are additional dimensions.
  /// \tparam Size  The type of the zero dimension size.
  /// \tparam Sizes The types of other dimension sizes.
  template <
    typename    Size ,
    typename... Sizes,
    arithmetic_size_enable_t<Dimensions, Size, Sizes...> = 0
  >
  TensorStorage(Size&& size_0, Sizes&&... sizes)
  : _storage{nullptr}, 
    _dim_sizes{
      static_cast<std::size_t>(size_0),
      static_cast<std::size_t>(sizes)...
   } {}

  //==--- [interface] ------------------------------------------------------==//

  /// Returns the size of the \p nth dimension for the tensor.
  /// \param dim The dimension to get the size of.
  ripple_host_device auto size(std::size_t dim) const -> std::size_t {
    return _dim_sizes[dim];
  }

  /// Returns the total number of elements in the tensor.
  ripple_host_device auto elements() const -> std::size_t {
    std::size_t elems = 1;
    unrolled_for<dimensions>([&] (auto dim) {
      elems *= _dim_sizes[dim];
    });
    return elems;
  }

 protected:
  storage_t   _storage;   //!< The storage for the tensor.
  dim_sizes_t _dim_sizes; //!< The dimension sizes for the tensor.
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_TENSOR_STORAGE_HPP


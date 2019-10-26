//==--- ripple/container/tensor_traits.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor_traits.hpp
/// \brief This file defines traits and forward declarations for tensor traits.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_TENSOR_TRAITS_HPP
#define FLUIDITY_CONTAINER_TENSOR_TRAITS_HPP

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// Declaration of a device tensor class which stores data on the device. this
/// will store the data in soa format if the type T is an Array<> type and the
/// value returned by ``ArrayTraits<T>::is_soa`` is true, otherwise the data is
/// stored with the elements contiguously.
///
/// to convert to the equivalent version of the tensor on the host-side, use
/// the ``as_host()`` function, and using the ``as_device()`` function will
/// return a copy of the tensor.
///
/// ~~~{.cpp}
/// auto dev_tensor = DeviceTensor<int, 1>(50);
/// 
/// // Get a host version:
/// auto host_tensor = dev_tensor.as_host();
/// 
/// // Get a new device version:
/// auto dev_tensor1 = dev_tensor.as_device();
/// auto dev_tensor2 = host_tensor.as_device();
/// ~~~
///
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions> class DeviceTensor;  

/// Declaration of a host tensor class which stores data on the device. This
/// will store the data in aos format if the type T is an Array<> type and the
/// value returned by ``ArrayTraits<T>::is_soa`` is true, otherwise the data is
/// stored with the elements contiguously.
///
/// to convert to the equivalent version of the tensor on the host-side, use
/// the ``as_host()`` function, and using the ``as_device()`` function will
/// return a copy of the tensor.
/// 
/// ~~~{.cpp}
/// auto host_tensor = HostTensor<int, 1>(50);
/// 
/// // Get a device version:
/// auto dev_tensor = host_tensor.as_device();
/// 
/// // Get a new host version:
/// auto host_tensor1 = dev_tensor.as_host();
/// auto host_tensor2 = host_tensor.as_host();
/// ~~~
/// 
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions> class HostTensor;

/// Decalration of a storage class for the tensor. This class stores the data
/// and dimension information for the tensor.
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions> class TensorStorage;

//==--- [traits] -----------------------------------------------------------==//

/// The TensorTraits class defines traits for a tensor.
/// \tparam Tensor The tensor to get the traits for.
/// \tparam Dims  The number of dimensions of the tensor.
template <typename Tensor> struct TensorTraits;

/// Specialization of the tensor traits for a host tensor.
/// \tparam T     The data type for the tensor.
/// \tparam Dims  The number of dimensions for the tensor.
template <typename T, std::size_t Dims>
struct TensorTraits<HostTensor<T, Dims>> {
  /// Defines the storage type for the tensor.
  using storage_t = array_traits_t<T>::storage_t;

  /// Defines the number of dimensions for the tensor.
  static constexpr auto dimensions = Dims;
};

/// Specialization of the tensor traits for a device tensor.
/// \tparam T     The data type for the tensor.
/// \tparam Dims  The number of dimensions for the tensor.
template <typename T, std::size_t Dims>
struct TensorTraits<DeviceTensor<T, Dims>> {
  /// Defines the storage type for the tensor.
  using storage_t = array_traits_t<T>::storage_t;

  /// Defines the number of dimensions for the tensor.
  static constexpr auto dimensions = Dims;
};

/// Traits for tensor storage.
/// \tparam T          The type of the data to store in the tensor.
/// \tparam Dimensions The number of dimensions for the tensor.
template <typename T, std::size_t Dimensions>
struct StorageTraits {
  /// Defines the type of the storage for the tensor. This uses the ArrayTraits
  /// to determine the type to store for the tensor, 
};

//==--- [aliases] ----------------------------------------------------------==//

/// Alias for a 1-dimensional host side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using host_tensor_1d_t = HostTensor<T, 1>;

/// Alias for a 1-dimensional device side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using device_tensor_1d_t = DeviceTensor<T, 1>;

/// Alias for a 2-dimensional host side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using host_tensor_2d_t = HostTensor<T, 2>;

/// Alias for a 2-dimensional device side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using device_tensor_2d_t = DeviceTensor<T, 2>;

/// Alias for a 3-dimensional host side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using host_tensor_3d_t = HostTensor<T, 3>;

/// Alias for a 3-dimensional device side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using device_tensor_3d_t = DeviceTensor<T, 3>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_TENSOR_TRAITS_HPP

//==--- ripple/core/container/tensor_traits.hpp ------------ -*- C++ -*- ---==//
//
//                                  Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor_triats.hpp
/// \brief This file defines traits classes for Tensors.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_TENSOR_TRAITS_HPP
#define RIPPLE_CONTAINER_TENSOR_TRAITS_HPP

#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

/*==--- [forward declarations] ---------------------------------------------==*/

/**
 * Forward declaration of Splitter struct which splits a tensor into units of
 * work for parallel processing.
 */
struct Splitter;

/**
 * Forward declaration of a tensor class for an N-dimensional container.
 *
 * \tparam T          The data type for the Tensor.
 * \tparam Dimensions The number of dimensions for the Tensor data.
 */
template <typename T, size_t Dimensions>
class Tensor;

/*==--- [tensor traits] ----------------------------------------------------==*/

/**
 * Tensor traits for a class of type T. This is the base case for when the type
 * T is *not* a tensor.
 *
 * \tparam T The type of the class to get the Tensor traits for.
 */
template <typename T>
struct TensorTraits {
  // clang-format off
  /** Defines the data type of the tensor data. */
  using Value          = T;
  /** Defines the type of the tensor global data iterator. */
  using Iterator       = std::void_t<>;
  /** Defines the type of the tensor shared data iterator. */
  using SharedIterator = std::void_t<>;


  // clang-format off
  /** Returns that the type T is not a tensor. */
  static constexpr bool   is_tensor  = false;
  /** Returns the number of dimensions of the non-tensor type T. */
  static constexpr size_t dimensions = 0;
  // clang-format on
};

/**
 * Tensor traits for a Tensor with data type T and Dims dimensions. This
 * specialization is for tensor types.
 *
 * \tparam T           The type of the tensor data.
 * \tparam Dimensions The number of dimensions for the tensor.
 */
template <typename T, size_t Dimensions>
struct TensorTraits<Tensor<T, Dimensions>> {
  // clang-format off
  /** Defines the data type of the tensor data. */
  using Value          = T;
  /** Defines the type of the iterator over the tensor in global memory. */
  using Iterator       = typename Block<T, Dimensions>::Iterator;
  /** Defines the type of the iterator over the tensor in shared memory. */
  using SharedIterator = typename Block<T, Dimensions>::SharedIterator;

  // clang-format off
  /** Returns that this is a Tensor. */
  static constexpr bool   is_tensor  = true;
  /** Returns the number of dimensions for the tensor. */
  static constexpr size_t dimensions = Dimensions;
  // clang-format on.
};

/*==--- [aliases] ----------------------------------------------------------==*/

/**
 * Alias for a 1D tensor.
 * \tparam T The data type for the tensor.
 */
template <typename T>
using Tensor1d = Tensor<T, 1>;

/**
 * Alias for a 2D tensor.
 * \tparam T The data type for the tensor.
 */
template <typename T>
using Tensor2d = Tensor<T, 2>;

/**
 * Alias for a 3D tensor.
 * \tparam T The data type for the tensor.
 */
template <typename T>
using Tensor3d = Tensor<T, 3>;

/**
 * Alias for the tensor traits for type T after decaying.
 * \tparam T The type to get the tensor traits for.
 */
template <typename T>
using tensor_traits_t = TensorTraits<std::decay_t<T>>;

/**
 * Returns true if the type T is a tensor.
 * \tparam T The type to determine if is a tensor.
 */
template <typename T>
static constexpr bool is_tensor_v = tensor_traits_t<T>::is_tensor;

/**
 * Defines a valid type if T is a tensor.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using tensor_enable_t = std::enable_if_t<is_tensor_v<T>, int>;

/**
 * Defines a valid type if T is not a tensor.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_tensor_enable_t = std::enable_if_t<!is_tensor_v<T>, int>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_TENSOR_TRAITS_HPP
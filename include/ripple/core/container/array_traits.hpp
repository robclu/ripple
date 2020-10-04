//==--- ripple/core/container/array_traits.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array_traits.hpp
/// \brief This file defines forward declarations and traits for arrays.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_ARRAY_TRAITS_HPP
#define RIPPLE_CONTAINER_ARRAY_TRAITS_HPP

#include <ripple/core/storage/storage_layout.hpp>
#include <ripple/core/utility/number.hpp>
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

/*==--- [forward declarations] ---------------------------------------------==*/

/**
 * The Array class defines a static interface for array types -- essentially
 * types which have contiguous storage and can be accessed with the index
 * operator.
 *
 * The implementation is provided by the template type Impl.
 * \tparam Impl The implementation of the array interface.
 */
template <typename Impl>
struct Array;

/**
 * This is an implementation class for a statically sized vector class with a
 * flexible storage layout.
 *
 * \note This is an implementation class, there are aliases for vectors,
 *       \sa Vector, \sa Vec.
 *
 * \tparam T      The type of the data for the vector.
 * \tparam Size   The size of the vector.
 * \tparam Layout The storage layout of the vector.
 */
template <typename T, typename Size, typename Layout = ContiguousOwned>
struct VecImpl;

/*==--- [traits] -----------------------------------------------------------==*/

/**
 * This structs defines traits for arrays. This implementation is the default
 * implementation and is for a type which is not an array. It needs to be
 * specialzied for any type which does implement the interface.
 *
 * \tparam T The type to get the array traits for.
 */
template <typename T>
struct ArrayTraits {
  // clang-format off
  /** The value type for the array. */
  using Value  = std::decay_t<T>;
  /** Defines the type of the layout for the array. */
  using Layout = ContiguousOwned;
  /**  Defines the type for an array of type T */
  using Array  = VecImpl<Value, Num<1>, Layout>;

  /** Returns the number of elements in the array.   */
  static constexpr auto size = 1;
  // clang-format on
};

/**
 * Specialization of the ArrayTraits class for the VecImpl class.
 *
 * \tparam T      The type of the data for the vec.
 * \tparam Size   The size  of the vector.
 * \tparam Layout The storage layout of the vector.
 */
template <typename T, typename Size, typename LayoutType>
struct ArrayTraits<VecImpl<T, Size, LayoutType>> {
  // clang-format off
  /** The value type stored in the array. */
  using Value  = std::decay_t<T>;
  /** Defines the type of the layout for the array. */
  using Layout = LayoutType;
  /** Defines the type of an array of the value type. */
  using Array  = VecImpl<Value, Size, Layout>;

  /** Returns the number of elements in the array.  */
  static constexpr auto size = Size::value;
  // clang-format on
};

/**
 * Specialization of the ArrayTraits class for the any class which implements
 * the array interface.
 *
 * \tparam Impl The implementation type of the array interface.
 */
template <typename Impl>
struct ArrayTraits<Array<Impl>> {
 private:
  /** Defines the type of the implementation traits. */
  using Traits = ArrayTraits<Impl>;

 public:
  // clang-format off
  /** Defines the value type of the array. */
  using Value  = typename Traits::Value;
  /** Defines the type of the layout for the array. */
  using Layout = typename Traits::Layout;
  /** Defines the type of an array of the value type. */
  using Array  = typename Traits::Array;

  /** Returns the number of elements in the array. */
  static constexpr auto size = Traits::size;
  // clang-format on
};

/*==--- [aliases & constants] ----------------------------------------------==*/

/**
 * Alias for a vector to store data of type T with Size elements.
 *
 * \note This uses the default layout type for the vector.
 *
 * \tparam T     The type to store in the vector.
 * \tparam Size  The size of the vector.
 */
template <typename T, size_t Size>
using Vec = VecImpl<T, Num<Size>>;

/**
 * Alias for a vector to store data of type T with Size elements with the given
 * Layout.
 *
 * \tparam T      The type to store in the vector.
 * \tparam Size   The size of the vector.
 * \tparam Layout The layout to store the vector data.
 */
template <typename T, size_t Size, typename Layout>
using Vector = VecImpl<T, Num<Size>, Layout>;

/**
 * Defines an alias to create a contiguous 1D vector of type T.
 * \tparam T The type of the data for the vector.
 */
template <typename T>
using Vec1d = VecImpl<T, Num<1>>;

/**
 * Defines an alias to create a contiguous 2D vector of type T.
 * \tparam T The type of the data for the vector.
 */
template <typename T>
using Vec2d = VecImpl<T, Num<2>>;

/**
 * Defines an alias to create a contiguous 3D vector of type T.
 * \tparam T The type of the data for the vector.
 */
template <typename T>
using Vec3d = VecImpl<T, Num<3>>;

/**
 * Gets the array traits for the type T.
 * \tparam T The type to get the array traits for.
 */
template <typename T>
using array_traits_t = ArrayTraits<std::decay_t<T>>;

/**
 * Returns true if the template parameter is an Array.
 * \tparam T The type to determine if is an array.
 */
template <typename T>
static constexpr bool is_array_v =
  std::is_base_of_v<Array<std::decay_t<T>>, std::decay_t<T>>;

/**
 * Defines a fallback vector type based on the type T.
 * \tparam T The type to base the fallback on.
 */
template <typename T, typename Traits = array_traits_t<T>>
using VecFallback =
  VecImpl<typename Traits::Value, Num<Traits::size>, ContiguousOwned>;

/**
 * Returns an implementation type which is copyable and trivially constructable
 * between implementations ImplA and ImplB. This will first check the valididy
 * of ImplA, and then of ImplB.
 *
 *  An implementation is valid if the storage type of the implementation is
 * **not** a pointer (since the data for the pointer will then need to be
 * allocaed) and **is** trivially constructible.
 *
 * If neither ImplA nor ImplB are valid, then this will default to using the
 * Vec type with a value type of ImplA and a the size of ImplA.
 *
 * \tparam ImplA The type of the first array implementation.
 * \tparam ImplB The type of the second array implementation.
 */
template <
  typename ImplA,
  typename ImplB,
  typename LayoutA  = typename array_traits_t<ImplA>::Layout,
  typename LayoutB  = typename array_traits_t<ImplB>::Layout,
  bool ValidityA    = std::is_same_v<LayoutA, ContiguousOwned>,
  bool ValidityB    = std::is_same_v<LayoutB, ContiguousOwned>,
  typename Fallback = VecFallback<ImplA>>
using array_impl_t = std::conditional_t<
  ValidityA,
  ImplA,
  std::conditional_t<ValidityB, ImplB, Fallback>>;

/*==--- [enables] ----------------------------------------------------------==*/

/**
 * Defines a valid type if the type T is either the same as the value type of
 * the array implementation Impl, or convertible to the value type.
 *
 * \tparam T    The type to base the enable on.
 * \tparam Impl The array implementation to get the value type from.
 */
template <
  typename T,
  typename Impl,
  typename Type  = std::decay_t<T>,
  typename Value = typename ArrayTraits<Impl>::Value>
using array_value_enable_t = std::enable_if_t<
  (std::is_same_v<Type, Value> ||
   std::is_convertible_v<Type, Value>)&&!is_array_v<Type>,
  int>;

/**
 * Defines a valid type if the type T is an array.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using array_enable_t = std::enable_if_t<is_array_v<T>, int>;

/**
 * Defines a valid type if the type T is _not_ an array.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_array_enable_t = std::enable_if_t<!is_array_v<T>, int>;

/**
 * Defines a valid type if the type T is an array and the size of the array is
 * Size.
 * \tparam T     The type to check if is an array.
 * \tparam Size  The size the array must have.
 */
template <typename T, size_t Size>
using array_size_enable_t =
  std::enable_if_t<is_array_v<T> && (array_traits_t<T>::size == Size), int>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_ARRAY_TRAITS_HPP

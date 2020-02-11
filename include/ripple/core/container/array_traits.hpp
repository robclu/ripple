//==--- ripple/core/container/array_traits.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
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

//==--- [forward declarations] ---------------------------------------------==//

/// The Array class defines an interface to which all specialized
/// implementations must conform. The implementation is provided by
/// the template type Impl.
/// \tparam Impl The implementation of the array interface.
template <typename Impl> struct Array;

/// The Vec class implements the Array interface to store data in AoS format.
/// \tparam T      The type of the data for the vector.
/// \tparam Size   The size of the vector.
/// \tparam Layout The storage layout of the vector.
template <
  typename T   ,
  typename Size, 
  typename Layout = contiguous_owned_t
> 
struct VecImpl;


//==--- [traits] -----------------------------------------------------------==//

/// The ArrayTraits structs defines traits that all classes which implement the
/// Array interface must implement. This is the default implementation, and
/// defines traits for a type T which does not implement the Array interface.
template <typename T>
struct ArrayTraits {
  /// The value type for the array.
  using value_t  = std::decay_t<T>;
  /// Defines the type of the layout for the array.
  using layout_t = contiguous_owned_t;
  /// Defines the type of the array.
  using array_t = VecImpl<value_t, Num<1>, layout_t>;

  /// Returns the number of elements in the array.  
  static constexpr auto size = 1;
};

/// Specialization of the ArrayTraits class for the Vec class.
/// \tparam T      The type of the data for the vec.
/// \tparam Size   The size  of the vector.
/// \tparam Layout The storage layout of the vector.
template <typename T, typename Size, typename Layout>
struct ArrayTraits<VecImpl<T, Size, Layout>> {
 public:
  /// The value type stored in the vector.
  using value_t  = std::decay_t<T>;
  /// The array type.
  using array_t  = VecImpl<T, Size, Layout>;
  /// Defines the type of the layout for the array.
  using layout_t = Layout;

  /// Returns the number of elements in the array.  
  static constexpr auto size = Size::value;
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
  /// Defines the value type of the array.
  using value_t  = typename impl_traits_t::value_t;
  /// The array type.
  using array_t  = typename impl_traits_t::array_t;
  /// Defines the type of the layout for the implementation.
  using layout_t = typename impl_traits_t::layout_t;

  /// Returns the number of elements in the array.  
  static constexpr auto size = impl_traits_t::size;
};

//==--- [aliases & constants] ----------------------------------------------==//

/// Alias for a vector to store data of type T with Size elements.
/// \tparam T     The type to store in the vector.
/// \tparam Size  The size of the vector.
template <typename T, size_t Size>
using Vec = VecImpl<T, Num<Size>>;

/// Alias for a vector to store data of type T with Size elements with the given
/// Layout.
/// \tparam T      The type to store in the vector.
/// \tparam Size   The size of the vector.
/// \tparam Layout The layout to store the vector data.
template <typename T, size_t Size, typename Layout>
using Vector = VecImpl<T, Num<Size>, Layout>;

/// Defines an alias to create a contiguous 1D vector of type T.
/// \tparam T The type of the data for the vector.
template <typename T>
using vec_1d_t = VecImpl<T, Num<1>>;

/// Defines an alias to create a contiguous 2D vector of type T.
/// \tparam T The type of the data for the vector.
template <typename T>
using vec_2d_t = VecImpl<T, Num<2>>;

/// Defines an alias to create a contiguous 3D vector of type T.
/// \tparam T The type of the data for the vector.
template <typename T>
using vec_3d_t = VecImpl<T, Num<3>>;

/// Gets the array traits for the type T.
/// \tparam T The type to get the array traits for.
template <typename T>
using array_traits_t = ArrayTraits<std::decay_t<T>>;

/// Returns true if the template parameter is an Array.
/// \tparam T The type to determine if is an array.
template <typename T>
static constexpr bool is_array_v =
  std::is_base_of_v<Array<std::decay_t<T>>, std::decay_t<T>>;

/// Returns an implementation type which is copyable and trivially constructable
/// between implementations ImplA and ImplB. This will first check the valididy
/// of ImplA, and then of ImplB.
///
/// An implementation is valid if the storage type of the implementation is
/// **not** a pointer (since the data for the pointer will then need to be
/// allocaed) and **is** trivially constructible.
///
/// If neither ImplA nor ImplB are valid, then this will default to using the
/// Vec type with a value type of ImplA and a the size of ImplA.
///
/// \tparam ImplA The type of the first array implementation.
/// \tparam ImplB The type of the second array implementation.
template <
  typename ImplA,
  typename ImplB,
  typename LayoutA   = typename array_traits_t<ImplA>::layout_t,
  typename LayoutB   = typename array_traits_t<ImplB>::layout_t,
  bool     ValidityA = std::is_same_v<LayoutA, contiguous_owned_t>,
  bool     ValidityB = std::is_same_v<LayoutB, contiguous_owned_t>,
  typename Fallback  = VecImpl<
    typename array_traits_t<ImplA>::value_t, 
    Num<array_traits_t<ImplA>::size>,
    contiguous_owned_t
  >
>
using array_impl_t = std::conditional_t<
  ValidityA, ImplA, std::conditional_t<ValidityB, ImplB, Fallback>   
>;

//==--- [enables] ----------------------------------------------------------==//

/// Defines a valid type if the type T is either the same as the value type of
/// the array implementation Impl, or convertible to the value type.
/// \tparam T    The type to base the enable on.
/// \tparam Impl The array implementation to get the value type from.
template <
  typename T,
  typename Impl,
  typename Type  = std::decay_t<T>,
  typename Value = typename ArrayTraits<Impl>::value_t
>
using array_value_enable_t = std::enable_if_t<
  std::is_same_v<Type, Value> || std::is_convertible_v<Type, Value>, int
>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_ARRAY_TRAITS_HPP

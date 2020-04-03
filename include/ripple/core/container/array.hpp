//==--- ripple/core/core/container/array.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  array.hpp
/// \brief This file defines an interface for arrays.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_ARRAY_HPP
#define RIPPLE_CONTAINER_ARRAY_HPP

#include "array_traits.hpp"
#include <ripple/core/algorithm/unrolled_for.hpp>

namespace ripple {

/// The Array class defines an interface to which all specialized
/// implementations must conform. The implementation is provided by the template
/// type Impl.
/// \tparam Impl The implementation of the array interface.
template <typename Impl>
struct Array {
 private:
  // Defines the type of the implementation.
  using impl_t = Impl;

 public:
  /// Returns the value at position \p i in the array.
  /// \param i The index of the element to return.
  ripple_host_device constexpr auto operator[](size_t i) {
    return impl()->operator[](i);  
  }

  /// Returns the value at position \p i in the array.
  /// \param i The index of the element to return.
  ripple_host_device constexpr auto operator[](size_t i) const {
    return impl()->operator[](i);  
  }
    
  /// Returns the number of elements in the array.
  ripple_host_device constexpr auto size() const -> size_t {
    return array_traits_t<impl_t>::size;
  }

  //==--- [operator {+,+=} overloads] --------------------------------------==//

  /// Overload of operator+= to add each element of array \p a to each element
  /// of this array. If the sizes of the arrays are different, this will cause a
  /// compile time error.
  /// \param  a     The array to add with.
  /// \tparam ImplA The implementation type of the addition array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator+=(const Array<ImplA>& a) 
  -> impl_t& {
    assert_size_match<ImplA>();
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) += a[i];
    });
    return *impl();
  }

  /// Overload of operator+= to add the value \p val to each element of this
  /// array. If the type T cannot be converted to the value type of the array
  /// then this will cause a compile time error.
  /// \param  val  The value to add to each element of the array.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator+=(T val) -> impl_t& { 
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) += static_cast<value_t>(val);
    });
    return *impl();
  }

  /// Overload of operator+ to add each element of array \p a to each element
  /// in this array, returning a new array with the same implementation as 
  /// either this array (if copyable), the second array (if copyable), or the
  /// fallback type.
  /// \param  a     The array for the addition.
  /// \tparam ImplA The implementation type of the other array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator+(const Array<ImplA>& a) const
  -> array_impl_t<impl_t, ImplA> {
    using res_impl_t      = array_impl_t<impl_t, ImplA>;
    auto           result = res_impl_t();
    constexpr auto size   = array_traits_t<res_impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) + a[i];
    });
    return result;
  }

  /// Overload of operator+ to add the value \p val to each element of this
  /// array, returning a new array with the same type as this array, or the
  /// fallback type. If the type T cannot be converted to the value type of the
  /// array then this will cause a compile time error.
  /// \param  val  The value to add to each element of the array.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator+(T val) const
  -> array_impl_t<impl_t, impl_t> { 
    using res_impl_t    = array_impl_t<impl_t, impl_t>;
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<res_impl_t>::size;
    auto result         = res_impl_t();
    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) + static_cast<value_t>(val);
    });
    return result;
  }

  //==--- [operator {-,-=} overloads] --------------------------------------==//

  /// Overload of operator-= to subtract each element of array \p a from each
  /// element of this array. If the sizes of the arrays are different, this will
  /// cause a compile time error.
  /// \param  a     The array to subtract with.
  /// \tparam ImplA The implementation type of the subtraction array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator-=(const Array<ImplA>& a) 
  -> impl_t& {
    assert_size_match<ImplA>();
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) -= a[i];
    });
    return *impl();
  }

  /// Overload of operator-= to subtract the value \p val from each element of
  /// this array. If the type T cannot be converted to the value type of the
  /// array then this will cause a compile time error.
  /// \param  val  The value to subtract from each element of the array.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator-=(T val) -> impl_t& { 
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) -= static_cast<value_t>(val);
    });
    return *impl();
  }

  /// Overload of operator- to subtract each element of array \p a from each
  /// element in this array, returning a new array with the same implementation
  /// type as the this array (if copyable), the second array (if copyable), or
  /// th fallback type.
  /// \param  a     The array for the subtraction.
  /// \tparam ImplA The implementation type of the subtraction array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator-(const Array<ImplA>& a) const
  -> array_impl_t<impl_t, ImplA> {
    using res_impl_t      = array_impl_t<impl_t, ImplA>;
    auto           result = res_impl_t();
    constexpr auto size   = array_traits_t<res_impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) - a[i];
    });
    return result;
  }

  /// Overload of operator- to subtract the value \p val from each element of
  /// this array, returning a new array with the same type as this array, or the
  /// fallback type. If the type T cannot be converted to the value type of the
  /// array then this will cause a compile time error.
  /// \param  val  The value to subtract from each element of the array.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator-(T val) const
  -> array_impl_t<impl_t, impl_t> { 
    using res_impl_t    = array_impl_t<impl_t, impl_t>;
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<res_impl_t>::size;
    auto result         = res_impl_t();
    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) - static_cast<value_t>(val);
    });
    return result;
  }

  //==--- [operator {*,*=} overloads] --------------------------------------==//

  /// Overload of operator*= to multiply each element of array \p a with each
  /// element of this array. If the sizes of the arrays are different, this will
  /// cause a compile time error.
  /// \param  a     The array to multiply with.
  /// \tparam ImplA The implementation type of the multiplication array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator*=(const Array<ImplA>& a) 
  -> impl_t& {
    assert_size_match<ImplA>();
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) *= a[i];
    });
    return *impl();
  }

  /// Overload of operator*= to multiply the value \p val with each element of
  /// this array. If the type T cannot be converted to the value type of the
  /// array then this will cause a compile time error.
  /// \param  val  The value to multiply with each element of the array.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator*=(T val) -> impl_t& { 
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) *= static_cast<value_t>(val);
    });
    return *impl();
  } 

  /// Overload of operator* to multiply each element of array \p a with each
  /// element in this array, returning a new array with the same implementation
  /// type as this array (if copyable), the second array (if copyable), or the 
  /// fallback type.
  /// \param  a     The array to multiply with.
  /// \tparam ImplA The implementation type of the multiplication array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator*(const Array<ImplA>& a) const
  -> array_impl_t<impl_t, ImplA> {
    using res_impl_t      = array_impl_t<impl_t, ImplA>;
    auto           result = res_impl_t();
    constexpr auto size   = array_traits_t<res_impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) * a[i];
    });
    return result;
  }

  /// Overload of operator* to multiply the value \p val with each element of
  /// this array, returning a new array with the same type as this array, or the
  /// fallback type. If the type T cannot be converted to the value type of the
  /// array then this will cause a compile time error.
  /// \param  val  The value to multiply with each element of the array.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator*(T val) const
  -> array_impl_t<impl_t, impl_t> { 
    using res_impl_t    = array_impl_t<impl_t, impl_t>;
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<res_impl_t>::size;
    auto result         = res_impl_t();

    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) * static_cast<value_t>(val);
    });
    return result;
  }

  //==--- [operator {/,/=} overloads] --------------------------------------==//

  /// Overload of operator*= to multiply each element of array \p a with each
  /// element of this array. If the sizes of the arrays are different, this will
  /// cause a compile time error.
  /// \param  a     The array to multiply with.
  /// \tparam ImplA The implementation type of the multiplication array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator/=(const Array<ImplA>& a) 
  -> impl_t& {
    assert_size_match<ImplA>();
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) /= a[i];
    });
    return *impl();
  }

  /// Overload of operator/= to divide the value \p val with each element of
  /// this array. If the type T cannot be converted to the value type of the
  /// array then this will cause a compile time error.
  /// \param  val  The value to divide each element of the array by.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator/=(T val) -> impl_t& { 
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      impl()->operator[](i) /= static_cast<value_t>(val);
    });
    return *impl();
  } 

  /// Overload of operator/ to divide each element of array \p a with each
  /// element in this array, returning a new array with the same implementation
  /// type as this array (if copyable), the second array (if copyable), or the 
  /// fallback type.
  /// \param  a     The array to divide by.
  /// \tparam ImplA The implementation type of the division array.
  template <typename ImplA>
  ripple_host_device constexpr auto operator/(const Array<ImplA>& a) const
  -> array_impl_t<impl_t, ImplA> {
    using res_impl_t      = array_impl_t<impl_t, ImplA>;
    auto           result = res_impl_t();
    constexpr auto size   = array_traits_t<res_impl_t>::size;
    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) / a[i];
    });
    return result;
  }

  /// Overload of operator/ to divide each element of this array by the value
  /// \p val, returning a new array with the same type as this array, or the
  /// fallback type. If the type T cannot be converted to the value type of the
  /// array then this will cause a compile time error.
  /// \param  val  The value to multiply with each element of the array.
  /// \tparam T    The type of the value.
  template <typename T, array_value_enable_t<T, impl_t> = 0>
  ripple_host_device constexpr auto operator/(T val) const
  -> array_impl_t<impl_t, impl_t> { 
    using res_impl_t    = array_impl_t<impl_t, impl_t>;
    using value_t       = typename array_traits_t<impl_t>::value_t;
    constexpr auto size = array_traits_t<res_impl_t>::size;
    auto result         = res_impl_t();
    unrolled_for_bounded<size>([&] (auto i) {
      result[i] = impl()->operator[](i) / static_cast<value_t>(val);
    });
    return result;
  }
    
 private:
  /// Returns a pointer to the implementation of the interface.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }
    
  /// Returns a pointer to constant implementation of the interface.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Performs a compile assertation to check that the size of the array
  /// implemented by ImplA has the same nunber of elements that this array does.
  /// \tparam ImplA The implementation type of the array to check.
  template <typename ImplA>
  ripple_host_device constexpr auto assert_size_match() const -> void {
    constexpr auto size   = array_traits_t<Impl>::size;
    constexpr auto size_a = array_traits_t<ImplA>::size;
    static_assert(
      size == size_a, 
      "Cannot perform operations on arrays with different number of elements!"    
    );
  }
};

//==--- [operator overloads] -----------------------------------------------==//

/// Overload of operator+ to add the scalar value \p val with each element of
/// the array \p a, returning a new array with an implementation type of \p a if
/// copyable, or the fallback type.
/// \param  val   The value to add with each element of the array.
/// \param  a     The array to add with the value.
/// \tparam T     The type of the scalar.
/// \tparam Impl  The implementation type of the array.
template <typename T, typename Impl, array_value_enable_t<T, Impl> = 0>
ripple_host_device constexpr auto operator+(T val, const Array<Impl>& a)
-> array_impl_t<Impl, Impl> {
  using impl_t = array_impl_t<Impl, Impl>;
  using value_t = typename array_traits_t<impl_t>::value_t;
  using type_t  = std::decay_t<T>;

  static_assert(
    std::is_same_v<type_t, value_t> || std::is_convertible_v<type_t, value_t>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!"
  );

  auto           result = impl_t();
  constexpr auto size   = array_traits_t<impl_t>::size;
  unrolled_for_bounded<size>([&] (auto i) {
    result[i] = static_cast<value_t>(val) + a[i];
  });
  return result;
}

/// Overload of operator- to subtract each element of the array \p a from the
/// scalar value \p val, returning a new array with an implementation type of
/// \p a if copyable, or the fallback type.
/// \param  val   The value to subtract from.
/// \param  a     The array to subtract with.
/// \tparam T     The type of the scalar.
/// \tparam Impl  The implementation type of the array.
template <typename T, typename Impl, array_value_enable_t<T, Impl> = 0>
ripple_host_device constexpr auto operator-(T val, const Array<Impl>& a)
-> array_impl_t<Impl, Impl> {
  using impl_t = array_impl_t<Impl, Impl>;
  using value_t = typename array_traits_t<impl_t>::value_t;
  using type_t  = std::decay_t<T>;

  static_assert(
    std::is_same_v<type_t, value_t> || std::is_convertible_v<type_t, value_t>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!"
  );

  auto           result = impl_t();
  constexpr auto size   = array_traits_t<impl_t>::size;
  unrolled_for_bounded<size>([&] (auto i) {
    result[i] = static_cast<value_t>(val) - a[i];
  });
  return result;
}

/// Overload of operator* to multiply each element of the array \p a with the
/// scalar value \p val, returning a new array with an implementation type of
/// \p a if copyable, or the fallback type.
/// \param  val   The value to multiply with the arry..
/// \param  a     The array to multiply with the scalar.
/// \tparam T     The type of the scalar.
/// \tparam Impl  The implementation type of the array.
template <typename T, typename Impl, array_value_enable_t<T, Impl> = 0>
ripple_host_device constexpr auto operator*(T val, const Array<Impl>& a)
-> array_impl_t<Impl, Impl> {
  using impl_t = array_impl_t<Impl, Impl>;
  using value_t = typename array_traits_t<impl_t>::value_t;
  using type_t  = std::decay_t<T>;

  static_assert(
    std::is_same_v<type_t, value_t> || std::is_convertible_v<type_t, value_t>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!"
  );

  auto           result = impl_t();
  constexpr auto size   = array_traits_t<impl_t>::size;
  unrolled_for_bounded<size>([&] (auto i) {
    result[i] = static_cast<value_t>(val) * a[i];
  });
  return result;
}

/// Overload of operator/ to divide the scalar value \p val by each element of
/// the array \p a, returning a new array with an implementation type of \p a
/// if copyable, or the fallback type.
/// \param  val   The value to divide.
/// \param  a     The array to divide by.
/// \tparam T     The type of the scalar.
/// \tparam Impl  The implementation type of the array.
template <typename T, typename Impl, array_value_enable_t<T, Impl> = 0>
ripple_host_device constexpr auto operator/(T val, const Array<Impl>& a)
-> array_impl_t<Impl, Impl> {
  using impl_t = array_impl_t<Impl, Impl>;
  using value_t = typename array_traits_t<impl_t>::value_t;
  using type_t  = std::decay_t<T>;

  static_assert(
    std::is_same_v<type_t, value_t> || std::is_convertible_v<type_t, value_t>,
    "Cannot perform operations on an array with a type which is not the "
    "value type, or convertible to the value type!"
  );

  auto           result = impl_t();
  constexpr auto size   = array_traits_t<impl_t>::size;
  unrolled_for_bounded<size>([&] (auto i) {
    result[i] = static_cast<value_t>(val) / a[i];
  });
  return result;
}

} // namespace ripple

#endif // RIPPLE_CONTAINER_ARRAY_HPP

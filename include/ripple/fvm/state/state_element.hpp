//==--- ripple/fvm/state/state_element.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_element.hpp
/// \brief This file defines a type which represents an element for a state.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STATE_STATE_ELEMENT_HPP
#define RIPPLE_STATE_STATE_ELEMENT_HPP

namespace ripple::fv {

/// The StateElement type is a utility type which can be used to create user
/// defined literals for state elelemts so that components of the state can be
/// set more clearly.
/// \tparam Chars The characters for the name of the element.
template <typename C, C... Chars>
struct StateElement {
  double value = 0.0;                 //!< The value for the element.
  char   name[sizeof...(Chars) + 1];  //!< The name of the element.
    
  /// Default constructor which is enabled so that we can use decltype on an
  /// empty StateElement to determine the type.
  ripple_host_device constexpr StateElement()
  : name{Chars..., '\0'} {}

  /// Constructor which sets the value of the component to \p v.
  /// \param v The value for the component.
  ripple_host_device constexpr StateElement(double v)
  : value(v), name{Chars..., '\0'} {}
    
  /// Overload of unary operator - to allow negation.
  constexpr auto operator-() const -> StateElement<C, Chars...> {
    return StateElement<C, Chars...>(-value);
  }
};

namespace detail {

/// Class to determine if the type T is a StateElement type. This is the
/// default implementation for the case that the type T is not a state element.
/// \tparam T The type to determine if is a state element.
template <typename T>
struct IsStateElement {
  /// Returns that the type is not a state element type.
  static constexpr auto value = false;
};

/// Specialization for a StateElement type.
/// \tparam Char  The type of the characters for the element name.
/// \tparam Chars The characters which name the element.
template <typename Char, Char... Chars>
struct IsStateElement<StateElement<Char, Chars...>> {
  /// Returns that the type is a state element.
  static constexpr auto value = true;
};

} // namespace detail

/// Returns true if the type T is a state element, otherwise it returns false.
/// \tparam T The type to determine if is a state element.
template <typename T>
static constexpr auto is_state_element_v = 
  detail::IsStateElement<std::decay_t<T>>::value;

/// Utility function which can be used with declval to easily define new state
/// element type aliases, for example:
/// \begin{code}
///   using rho_t = decltype("rho"_state);
/// \end{code}
/// will create a new StateElement with the name "rho".
/// \tparam Char The type of the name characters
/// \tparam C    The characters in the name.
template <typename Char, Char... C>
constexpr auto operator "" _element() -> StateElement<Char, C...> {
  return StateElement<Char, C...>();
}

//==--- [aliases] ----------------------------------------------------------==//

namespace state {

/// Defines an element for the density.
using rho_t      = decltype("rho"_element);
/// Defines an element for the pressure.
using pressure_t = decltype("pressure"_element);
/// Defines an element for the x velocity.
using v_x_t      = decltype("v_x"_element);
/// Defines an element for the y velocity.
using v_y_t      = decltype("v_y"_element);
/// Defines an element for the z velocity.
using v_z_t      = decltype("v_z"_element);

/// User defined literal to create a density state element with value \p v.
/// \param v The value of the density.
auto operator "" _rho(long double v) -> rho_t {
  return rho_t(v);
}

/// User defined literal to create a pressure state element with value \p v.
/// \param v The value of the pressure.
auto operator "" _p(long double v) -> pressure_t {
  return pressure_t(v);
}

/// User defined literal to create an x velocity state element with value \p v.
/// \param v The value of the velocity.
auto operator "" _v_x(long double v) -> v_x_t {
  return v_x_t(v);
}

/// User defined literal to create an y velocity state element with value \p v.
/// \param v The value of the velocity.
auto operator "" _v_y(long double v) -> v_y_t {
  return v_y_t(v);
}

/// User defined literal to create an z velocity state element with value \p v.
/// \param v The value of the velocity.
auto operator "" _v_z(long double v) -> v_z_t {
  return v_z_t(v);
}

} // namespace state
} // namespace ripple::fv

#endif // RIPPLE_STATE_STATE_ELEMENT_HPP


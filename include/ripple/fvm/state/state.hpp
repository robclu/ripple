//==--- ripple/fv/state/state.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state.hpp
/// \brief This file defines an implementation for a static interface for state
///        types.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FV_STATE_STATE_HPP
#define RIPPLE_FV_STATE_STATE_HPP

#include "state_traits.hpp"

namespace ripple::fv {

/// The State type defines a static interface for state types. This interface
/// provides the minimum required functionality for states, which should be
/// extended by more complex states.
/// \tparam Impl The implementation type of the interface.
template <typename Impl> 
class State {
  /// Defines the type of the implementation.
  using impl_t   = std::decay_t<Impl>;
  /// Defines the traits for the state.
  using traits_t = StateTraits<impl_t>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 public:
  /// Defines the value type for the state.
  using value_t    = typename traits_t::value_t;
  /// Defines the type of the flux vector.
  using flux_vec_t = typename traits_t::flux_vec_t;

   //==--- [density] --------------------------------------------------------==//
  
  /// Returns a reference to the density of the fluid.
  ripple_host_device auto rho() -> value_t& {
    return impl()->rho();
  }

  /// Returns a const reference to the density of the fluid.
  ripple_host_device auto rho() const -> const value_t& {
    return impl()->rho();
  }

  //==--- [rho * v] --------------------------------------------------------==//
  
  /// Returns a reference to the convervative type $\rho v_i$, for the component
  /// of  velocity in the \p dim dimension.
  /// \param  dim The dimension to get the component for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto rho_v(Dim&& dim) -> value_t& {
    return impl()->rho_v(std::forward<Dim>(dim));
  }

  /// Returns a const reference to the convervative type $\rho v_i$, for the 
  /// component of  velocity in the \p dim dimension.
  /// \param  dim The index of the dimension for the component for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto rho_v(Dim&& dim) const -> const value_t& {
    return impl()->rho_v(std::forward<Dim>(dim));
  }

  /// Returns a reference to the convervative type $\rho v_i$, for the component
  /// of  velocity in the Dim dimension. This overload computes the offset to
  /// the component at compile time.
  /// \tparam Dim The dimension to get the velocity for.
  template <size_t Dim>
  ripple_host_device auto rho_v() -> value_t& {
    return impl()->template rho_v<Dim>();
  }

  /// Returns a constant reference to the convervative type $\rho v_i$, for the
  /// component of  velocity in the Dim dimension. This overload computes the 
  /// offset to the component at compile time.
  /// \tparam Dim The dimension to get the velocity for.
  template <size_t Dim>
  ripple_host_device auto rho_v() const -> const value_t& {
    return impl()->template rho_v<Dim>();
  }

  //==--- [velocity] -------------------------------------------------------==//

  /// Returns the velocity of the fluid in the \p dim dimension.
  /// \param  dim The dimension to get the velocity for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto v(Dim&& dim) const -> value_t {
    return impl()->v(std::forward<Dim>(dim));
  }

  /// Returns the velocity of the fluid in the Dim dimension. This overlod
  /// computes the offset to the velocity component at compile time.
  /// \tparam Dim The dimension to get the velocity for.
  template <size_t Dim>
  ripple_host_device auto v() const -> value_t {
    return impl()->template v<Dim>();
  }

  /// Sets the velocity of the state in the \p dim dimension to \p value.
  /// \param  dim   The dimension to set the velocity for.
  /// \param  value The value to set the velocity to.
  /// \tparam Dim   The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto set_v(Dim&& dim, value_t value) -> void {
    impl()->set_v(std::forward<Dim>(dim), value);
  }

  /// Sets the velocity of the state in the Dim direction to \p value. This
  /// overload computes the offset to the velocity at compile time.
  /// \param  value The value to set the velocity to.
  /// \tparam Dim   The dimension to set the velocity in.
  template <size_t Dim>
  ripple_host_device auto set_v(value_t value) -> void {
    impl()->template set_v<Dim>(value);
  } 

  //==--- [energy] ---------------------------------------------------------==//
 
  /// Returns a reference to the energy of the fluid. 
  ripple_host_device auto energy() -> value_t& {
    return impl()->energy();
  }

  /// Returns a constant reference to the energy of the fluid.
  ripple_host_device auto energy() const -> const value_t& {
    return impl()->energy();
  }

  //==--- [pressure] -------------------------------------------------------==//
  
  /// Returns the pressure of the fluid, using the \p eos equation of state
  /// for the fluid.
  /// \param  eos     The equation of state to use to compute the pressure.
  /// \tparam EosImpl The implementation of the equation of state interface.
  template <typename EosImpl>
  ripple_host_device auto pressure(const Eos<EosImpl>& eos) const -> value_t {
    return impl()->pressure(eos);
  }

  /// Sets the pressure of the state. Since the pressure is not stored, this
  /// computes the energy required for the state to have the given pressure \p
  /// p.
  /// \param  p       The value of the pressure for the state.
  /// \param  eos     The equation of state to use to compute the pressure.
  /// \tparam EosImpl The implementation of the equation of state interface.
  template <typename EosImpl>
  ripple_host_device auto set_pressure(value_t p, const Eos<EosImpl>& eos)
  -> void {
    return impl()->set_pressure(p, eos);
  }

  //==--- [flux] -----------------------------------------------------------==//
  
  /// Returns the flux for the state, in dimension \p dim.
  /// \param  eos     The equation of state to use.
  /// \param  dim     The dimension to get the flux in.
  /// \tparam EosImpl The implementation of the eqaution of state.
  /// \tparam Dim     The type of the dimension specifier.
  template <typename EosImpl, typename Dim>
  ripple_host_device auto flux(const Eos<EosImpl>& eos, Dim&& dim) const 
  -> flux_vec_t {
    return impl()->flux(eos, std::forward<Dim>(dim));
  }

  //==--- [direct access] --------------------------------------------------==//

  /// Overload of operator[] to enable array functionality on the state. This
  /// returns a reference to the \p ith stored element.
  /// \param i The index of the element to return.
  ripple_host_device auto operator[](size_t i) -> value_t& {
    return impl()->operator[](i);
  }

  /// Overload of operator[] to enable array functionality on the state. This
  /// returns a constant reference to the \p ith stored element.
  /// \param i The index of the element to return.
  ripple_host_device auto operator[](size_t i) const -> const value_t& {
    return impl()->operator[](i);
  }
};

} // namespace ripple::fv

#endif // RIPPLE_FV_STATE_STATE_HPP


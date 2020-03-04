//==--- ripple/fv/state/fluid_state.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fluid_state.hpp
/// \brief This file defines an implementation for a fluid state.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FV_STATE_FLUID_STATE_HPP
#define RIPPLE_FV_STATE_FLUID_STATE_HPP

#include "state.hpp"
#include <ripple/fvm/eos/eos.hpp>
#include <ripple/core/container/array.hpp>
#include <ripple/core/container/vec.hpp>
#include <ripple/core/math/math.hpp>
#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/viz/printable/printable.hpp>
#include <ripple/viz/printable/printable_element.hpp>

namespace ripple::fv {

/// The FluidState type defines a class which stores state data which can be
/// used to solve equations invloving fluids.
///
/// This state stores the data in conservative form, and stores the data as
/// follows (for a contiguous storage format in 3D):
///
///   - density         | $\rho$
///   - energy          | $E$
///   - density * v_x   | $\rho_u$
///   - density * v_y   | $\rho_v$
///   - density * v_z   | $\rho_w$
///
/// Through benchmarking, this is the optimal storage format for the most
/// efficient computation of the flux for the state, along with most of the
/// tested finite volume schemes (FORCE, HLLC, ...).
///
/// It also allows for the primitive type to be accessed, however, conversion is
/// requirted to compute the pressure.
///
/// \tparam T      The data type for the state data.
/// \tparam Dims   The number of dimensions for the fluid.
/// \tparam Layout The layout type for the data.
template <typename T, typename Dims, typename Layout>
class FluidState : 
  public Array<FluidState<T, Dims, Layout>>,
  public State<FluidState<T, Dims, Layout>>,
  public viz::Printable<FluidState<T, Dims, Layout>> {
  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines the number of dimensions for the state.
  static constexpr auto dims     = size_t{Dims::value};
  /// Defines the total number of elements in the state.
  static constexpr auto elements = dims + 2;
  /// Defines the offset to the velocity components.
  static constexpr auto v_offset = 2;
  /// Defines a constexpt value of 1 which is used throught the state.
  static constexpr auto _1       = T{1};

  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of this state.
  using self_t       = FluidState;
  /// Defines the type of the traits for the state.
  using traits_t     = StateTraits<self_t>;
  /// Defines the value type of the state data.
  using value_t      = typename traits_t::value_t;
  /// Defines the type for the storage descriptor for the state. The descriptor
  /// needs to store two elements for the density and the pressure, and dims
  /// elements for the velocity componenets.
  using descriptor_t = StorageDescriptor<Layout, StorageElement<T, elements>>;
  /// Defines the type of the storage for the state.
  using storage_t    = typename descriptor_t::storage_t;

 public:
  //==--- [aliases] --------------------------------------------------------==//

  /// Defines the type of the flux vector.
  using flux_vec_t = typename traits_t::flux_vec_t;

  //==--- [construction] ---------------------------------------------------==//
  
  /// Default constructor which creates the state.
  ripple_host_device FluidState() {}

  /// Constructor to create the state from the storage type.
  /// \param storage The storage to create this state with.
  ripple_host_device FluidState(storage_t storage) : _storage(storage) {}

  /// Constructor to create the state from another fluid type with a potentially
  /// different storage layout, copying the data from the other storage to this
  /// one.
  /// \param  other The other fluid state to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device FluidState(const FluidState<T, Dims, OtherLayout>& other)
  : _storage(other._storage) {}

  /// Constructor to create the state from another fluid type with a potentially
  /// different storage layout, moving the data from the other storage to this
  /// one.
  /// \param  other The other fluid state to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device FluidState(FluidState<T, Dims, OtherLayout>&& other)
  : _storage(std::move(other._storage)) {}

  //==--- [operator overload] ----------------------------------------------==//
  
  /// Overload of copy assignment operator to set the state from another state
  /// of a potentially different layout type.
  /// \param  other The other fluid state to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device auto operator=(
    const FluidState<T, Dims, OtherLayout>& other
  ) -> self_t& {
    _storage = other._storage;
    return *this;
  }

  /// Overload of move  assignment operator to set the state from another state
  /// of a potentially different layout type.
  /// \param  other The other fluid state to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device auto operator=(FluidState<T, Dims, OtherLayout>&& other)
  -> self_t& {
    _storage = other._storage;
    return *this;
  }

  /// Overload of operator[] to enable array functionality on the state. This
  /// returns a reference to the \p ith stored element.
  /// \param i The index of the element to return.
  ripple_host_device auto operator[](size_t i) -> value_t& {
    return _storage.template get<0>(i);
  }

  /// Overload of operator[] to enable array functionality on the state. This
  /// returns a constant reference to the \p ith stored element.
  /// \param i The index of the element to return.
  ripple_host_device auto operator[](size_t i) const -> const value_t& {
    return _storage.template get<0>(i);
  }

  //==--- [size] -----------------------------------------------------------==//
  
  /// Returns the size of the state.
  ripple_host_device constexpr auto size() const -> size_t {
    return elements;
  }

  //==--- [dimensions] -----------------------------------------------------==//
  
  /// Returns the number of spatial dimensions for the state.
  ripple_host_device constexpr auto dimensions() const -> size_t {
    return dims;
  }

  //==--- [density] --------------------------------------------------------==//
  
  /// Returns a reference to the density of the fluid.
  ripple_host_device auto rho() -> value_t& {
    return _storage.template get<0, 0>();
  }

  /// Returns a const reference to the density of the fluid.
  ripple_host_device auto rho() const -> const value_t& {
    return _storage.template get<0, 0>();
  }

  //==--- [velocity] -------------------------------------------------------==//
  
  /// Returns a reference to the convervative type $\rho v_i$, for the component
  /// of  velocity in the \p dim dimension.
  /// \param  dim The dimension to get the component for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto rho_v(Dim&& dim) -> value_t& {
    return _storage.template get<0>(v_offset + static_cast<size_t>(dim));
  }

  /// Returns a const reference to the convervative type $\rho v_i$, for the 
  /// component of  velocity in the \p dim dimension.
  /// \param  dim The index of the dimension for the component for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto rho_v(Dim&& dim) const -> const value_t& {
    return _storage.template get<0>(v_offset + static_cast<size_t>(dim));
  }

  /// Returns a reference to the convervative type $\rho v_i$, for the component
  /// of  velocity in the Dim dimension. This overload computes the offset to
  /// the component at compile time.
  /// \tparam Dim The dimension to get the velocity for.
  template <size_t Dim>
  ripple_host_device auto rho_v() -> value_t& {
    static_assert((Dim < dims), "Dimension out of range!");
    return _storage.template get<0, v_offset + Dim>();
  }

  /// Returns a constant reference to the convervative type $\rho v_i$, for the
  /// component of  velocity in the Dim dimension. This overload computes the 
  /// offset to the component at compile time.
  /// \tparam Dim The dimension to get the velocity for.
  template <size_t Dim>
  ripple_host_device auto rho_v() const -> const value_t& {
    static_assert((Dim < dims), "Dimension out of range!");
    return _storage.template get<0, v_offset + Dim>();
  }

  /// Returns the velocity of the fluid in the \p dim dimension.
  /// \param  dim The dimension to get the velocity for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto v(Dim&& dim) const -> value_t {
    return rho_v(static_cast<size_t>(dim)) / rho();
  }

  /// Returns the velocity of the fluid in the Dim dimension. This overlod
  /// computes the offset to the velocity component at compile time.
  /// \tparam Dim The dimension to get the velocity for.
  template <size_t Dim>
  ripple_host_device auto v() const -> value_t {
    return rho_v<Dim>()/ rho();
  }

  /// Sets the velocity of the state in the \p dim dimension to \p value.
  /// \param  dim   The dimension to set the velocity for.
  /// \param  value The value to set the velocity to.
  /// \tparam Dim   The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device auto set_v(Dim&& dim, value_t value) -> void {
    rho_v(static_cast<size_t>(dim)) = value * rho();
  }

  /// Sets the velocity of the state in the Dim direction to \p value. This
  /// overload computes the offset to the velocity at compile time.
  /// \param  value The value to set the velocity to.
  /// \tparam Dim   The dimension to set the velocity in.
  template <size_t Dim>
  ripple_host_device auto set_v(value_t value) -> void {
    rho_v<Dim>() = value * rho();
  }

  //==--- [energy] ---------------------------------------------------------==//
 
  /// Returns a reference to the energy of the fluid. 
  ripple_host_device auto energy() -> value_t& {
    return _storage.template get<0, 1>();
  }

  /// Returns a constant reference to the energy of the fluid.
  ripple_host_device auto energy() const -> const value_t& {
    return _storage.template get<0, 1>();
  }

  //==--- [pressure] -------------------------------------------------------==//
  
  /// Returns the pressure of the fluid, using the \p eos equation of state
  /// for the fluid.
  /// \param  eos     The equation of state to use to compute the pressure.
  /// \tparam EosImpl The implementation of the equation of state interface.
  template <typename EosImpl>
  ripple_host_device auto pressure(const Eos<EosImpl>& eos) const -> value_t {
    auto rho_v_sq = rho_v_squared_sum();
    return (eos.adi() - _1) * (energy() - value_t{0.5} * rho_v_sq / rho());

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
    const auto rho_v_sq = rho_v_squared_sum();
    energy() = p / (eos.adi() - _1) + (value_t{0.5} * rho_v_sq / rho());
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
    flux_vec_t f;
    const auto p = pressure(eos);
    const auto v = rho_v(dim) / rho();

    f.template at<0>() = rho_v(dim);
    f.template at<1>() = v * (energy() + p);
    f[v_offset + dim]  = rho_v(dim) * v + p;

    // Set the remaining velocity components:
    size_t shift = 0;
    unrolled_for<dims - 1>([&] (auto d) {
      if (d == dim) shift++;

      f[v_offset + d + shift] = rho_v(d + shift) * v;
    });
    return f;
  }

  //==--- [printable interface] --------------------------------------------==//
  
  /// Returns true if the state has an element with the \p name which can be
  /// printed.
  /// \param name The name of the element to check for.
  ripple_host_only auto has_printable_element(const char* name) const
  -> bool {
    return printable_element(name) != viz::PrintableElement::not_found();
  }

  /// Returns a printable element with the name \p name, using the arguments \p
  /// eos 
  /// \param  name The name of the element to get.
  /// \param  args Optional arguments used to get the elements.
  /// \tparam Args The type of the arguments.
  template <typename... Args>
  ripple_host_device auto printable_element(const char* name, Args&&... args)
  const -> viz::PrintableElement {
    using namespace math;
    const auto scalar = viz::PrintableElement::AttributeKind::scalar;
    const auto vector = viz::PrintableElement::AttributeKind::vector;
    switch (hash(name)) {
      case "density"_hash:
        return viz::PrintableElement("density", scalar, rho());
      case "energy"_hash:
        return viz::PrintableElement("energy", scalar, energy());
      case "pressure"_hash: {
        if constexpr (sizeof...(Args) == 1) {
          return viz::PrintableElement("pressure", scalar, pressure(args...));
        } else {
          return viz::PrintableElement::not_found();
        }
      }
      case "velocity"_hash: {
        auto e = viz::PrintableElement("velocity", vector);
        unrolled_for<dims>([&] (auto d) {
          constexpr auto dim = size_t{d};
          const auto vel = this->template v<dim>();
          e.add_value(vel);
        });
        return e;
      }
      default:
        return viz::PrintableElement::not_found();
    }
  }
  
 private:
  storage_t _storage; //!< Storage for the state data.

  /// Returns the sum of the rho_v components squared.
  ripple_host_device auto rho_v_squared_sum() const -> value_t {
    auto v_sq = value_t{0};
    unrolled_for<dims>([&] (auto d) {
      constexpr auto dim = size_t{d};
      v_sq += rho_v<dim>() * rho_v<dim>();
    });
    return v_sq;
  }
};

} // namespace ripple::fv

#endif // RIPPLE_FV_STATE_FLUID_STATE_HPP


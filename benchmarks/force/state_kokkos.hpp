/**=--- ripple/benchmarks/force/state_kokkos.hpp ----------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  state_kokkos.hpp
 * \brief This file defines an implementation for a fluid state for Kokkos.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARK_FORCE_STATE_KOKKOS_HPP
#define RIPPLE_BENCHMARK_FORCE_STATE_KOKKOS_HPP

#include <ripple/math/math.hpp>

/**
 * The State type defines a class which stores state data which can be
 * used to solve equations invloving fluids.
 *
 * This state stores the following data (for a 3D state):
 *
 *   - density  | $\rho$
 *   - pressure | $p$
 *   - v_x      | $u$
 *   - v_y      | $v$
 *   - v_z      | $w$
 *
 * \tparam T      The data type for the state data.
 * \tparam Dims   The number of dimensions for the fluid.
 * \tparam Layout The layout type for the data.
 */
template <typename T, size_t Dims>
class State {
 public:
  //==--- [constants] ------------------------------------------------------==//

  // clang-format off
  /** Defines the number of dimensions for the state. */
  static constexpr size_t dims     = Dims;
  /** Defines the total number of elements in the state. */
  static constexpr size_t elements = dims + 2;
  /** Defines the offset to the velocity components. */
  static constexpr size_t v_offset = 2;
  /** Defines a constexpt value of 1 which is used throught the state. */
  static constexpr auto   one      = T{1};

  /** Defines the value type of the state data. */
  using ValueType   = std::remove_reference_t<T>;

  /*==--- [construction] ---------------------------------------------------==*/

  /** Default constructor which creates the state. */
  ripple_all State() noexcept {}

  /**
   * Constructor to set all state elements to \p value.
   * \param value The value to set all elements to.
   */
  ripple_all State(ValueType value) noexcept {
    for (size_t i = 0; i < elements; ++i) {
      data_[i] = value;
    }
  }

  /**
   * Constructor to create the state from another fluid type with a potentially
   * different storage layout, copying the data from the other storage to this
   * one.
   * \param  other The other fluid state to create this one from.
   * \tparam OtherLayout The storage layout of the other type.
   */
  ripple_all State(const State& other) noexcept {
    for (size_t i = 0; i < elements; ++i) {
      data_[i] = other.data_[i];
    }
  }

  /**
   * Constructor to create the state from an array type (such as for fluxes).
   * \param  arr       The array to set the data from.
   * \tparam ArrayImpl The type of the array implementation.
   */
  // template <typename ArrayImpl, ripple::array_enable_t<ArrayImpl> = 0>
  // ripple_all State(const ArrayImpl& arr) noexcept {
  //  copy(arr);
  //}

  /*==--- [operator overload] ----------------------------------------------==*/

  /**
   * Overload of the assignment operator to set all element of the state to
   * \p value.
   * \param value the value to set all the state data to.
   */
  ripple_all auto operator=(ValueType value) noexcept -> State& {
    for (size_t i = 0; i < elements; ++i) {
      data_[i] = value;
    }
    return *this;
  }

  /**
   * Overload of copy assignment operator to set the state from another state
   * of a potentially different layout type.
   * \param  other The other fluid state to create this one from.
   * \tparam OtherLayout The storage layout of the other type.
   */
  ripple_all auto operator=(const State& other) noexcept -> State& {
    for (size_t i = 0; i < elements; ++i) {
      data_[i] = other.data_[i];
    }
    return *this;
  }

  /**
   * Overload of operator[] to enable array functionality on the state. This
   * returns a reference to the \p ith stored element.
   * \param i The index of the element to return.
   */
  ripple_all auto operator[](size_t i) noexcept -> ValueType& {
    return data_[i];
  }

  /**
   * Overload of operator[] to enable array functionality on the state. This
   * returns a constant reference to the \p ith stored element.
   * \param i The index of the element to return.
   */
  ripple_all auto operator[](size_t i) const noexcept -> const ValueType& {
    return data_[i];
  }

  /**
   * Returns the size of the state.
   */
  ripple_all constexpr auto size() const noexcept -> size_t {
    return elements;
  }

  /**
   * Returns the number of spatial dimensions for the state.
   */
  ripple_all constexpr auto dimensions() const noexcept -> size_t {
    return dims;
  }

  /**
   * Returns a reference to the density of the fluid.
   */
  ripple_all auto rho() noexcept -> ValueType& {
    return data_[0];
  }

  /**
   * Returns a const reference to the density of the fluid.
   */
  ripple_all auto rho() const noexcept -> const ValueType& {
    return data_[0];
  }

  /*==--- [velocity]
   * -------------------------------------------------------==*/

  /**
   * Returns the density * velocity for the dimension.
   * \param  dim The index of the dimension for the component for.
   * \tparam Dim The type of the dimension specifier.
   */
  ripple_all auto rho_v(size_t dim) const noexcept -> ValueType {
    return v(dim) * rho();
  }

  /**
   * Returns the velocity of the fluid in the \p dim dimension.
   * \param  dim The dimension to get the velocity for.
   * \tparam Dim The type of the dimension specifier.
   */
  ripple_all auto v(size_t dim) noexcept -> ValueType& {
    return data_[v_offset + dim];
  }

  /**
   * Returns the velocity of the fluid in the \p dim dimension.
   * \param  dim The dimension to get the velocity for.
   * \tparam Dim The type of the dimension specifier.
   */
  ripple_all auto v(size_t dim) const noexcept -> const ValueType& {
    return data_[v_offset + dim];
  }

  /**
   * Sets the velocity of the state in the \p dim dimension to \p value.
   * \param  dim   The dimension to set the velocity for.
   * \param  value The value to set the velocity to.
   * \tparam Dim   The type of the dimension specifier.
   */
  ripple_all auto set_v(size_t dim, ValueType value) noexcept -> void {
    v(dim) = value;
  }

  /**
   * Returns a vector of the velocities.
   */
  //ripple_all auto vel() noexcept -> VelocityVec {
  //  ValueType vel[dims];
  //  for (size_t d = 0; d < dims; ++d) {
  //    vel[d] = v(d);
  //  }
  //  return vel;
  //}

  /*==--- [energy]
   * ---------------------------------------------------------==*/

  /**
   * Returns a reference to the energy of the fluid.
   */
  ripple_all auto energy() noexcept -> ValueType& {
    return data_[1];
  }

  /*
   * Returns a constant reference to the energy of the fluid.
   */
  ripple_all auto energy() const noexcept -> const ValueType& {
    return data_[1];
  }

  //==--- [pressure]
  //-------------------------------------------------------==//

  /**
   * Returns the pressure of the fluid, using the \p eos equation of state
   * for the fluid.
   * \param  eos     The equation of state to use to compute the pressure.
   * \tparam EosImpl The implementation of the equation of state interface.
   */
  template <typename EosImpl>
  ripple_all auto pressure(const EosImpl& eos) const noexcept -> ValueType {
    const auto v_sq = v_squared_sum();
    return (eos.adi() - one) * (energy() - ValueType{0.5} * rho() * v_sq);
  }

  /**
   * Sets the pressure of the state. Since the pressure is not stored, this
   * computes the energy required for the state to have the given pressure \p
   * p.
   * \param  p       The value of the pressure for the state.
   * \param  eos     The equation of state to use to compute the pressure.
   * \tparam EosImpl The implementation of the equation of state interface.
   */
  template <typename EosImpl>
  ripple_all auto
  set_pressure(ValueType p, const EosImpl& eos) noexcept -> void {
    const auto v_sq = v_squared_sum();
    energy()        = p / (eos.adi() - one) + (ValueType{0.5} * v_sq * rho());
  }

  /*==--- [flux]
   * -----------------------------------------------------------==*/

  /**
   * Returns the flux for the state, in dimension \p dim.
   * \param  eos     The equation of state to use.
   * \param  dim     The dimension to get the flux in.
   * \tparam EosImpl The implementation of the eqaution of state.
   * \tparam Dim     The type of the dimension specifier.
   */
  template <typename EosImpl>
  ripple_all auto flux(const EosImpl& eos, size_t dim) const noexcept -> State {
    State      f;
    const auto p = pressure(eos);

    f[0]              = rho_v(dim);
    f[1]              = v(dim) * (energy() + p);
    f[v_offset + dim] = rho_v(dim) * v(dim) + p;

    // Set the remaining velocity components:
    size_t shift = 0;
    for (size_t d = 0; d < dims - 1; ++d) {
      if (d == dim) {
        shift++;
      }
      f[v_offset + d + shift] = rho_v(d + shift) * v(dim);
    }
    return f;
  }

  /*==--- [wavespeed] ------------------------------------------------------==*/

  /**
   * Returns the wavespeed for the state.
   * \param  eos     The equation of state for the state data.
   * \tparam EosImpl The implemenation of the equation of state interface.
   */
  template <typename EosImpl>
  ripple_all auto wavespeed(const EosImpl& eos) const noexcept -> ValueType {
    ValueType s = std::abs(rho_v(0));
    for (size_t d = 0; d < dims - 1; ++d) {
      const auto dim = d + 1;
      s              = std::max(s, std::abs(rho_v(dim)));
    }
    return s / rho() + eos.sound_speed(*this);
  }

 private:
  ValueType data_[elements]; //!< Storage for the state data.

  /**
   * Returns the sum of the rho_v components squared.
   */
  ripple_all auto rho_v_squared_sum() const noexcept -> ValueType {
    ValueType v_sq = ValueType{0};
    for (size_t d = 0; d < dims; ++d) {
      v_sq += rho_v(d) * rho_v(d);
    }
    return v_sq;
  }

  /**
   * Returns the sum of the velocity components squared.
   */
  ripple_all auto v_squared_sum() const noexcept -> ValueType {
    ValueType v_sq = ValueType{0};
    for (size_t d = 0; d < dims; ++d) {
      v_sq += v(d) * v(d);
    }
    return v_sq;
  }

  /**
   * Copies the data from the \p arr array type to this state.
   * \param  arr The array type to copy data from.
   * \tparam Arr The type of the array.
   */
  template <typename Arr>
  ripple_all auto copy(const Arr& arr) noexcept -> void {
    for (size_t i = 0; i < elements; ++i) {
      data_[i] = arr[i];
    }
  }
};

/*==--- [operator overloading] ---------------------------------------------==*/

template <typename T, size_t Dims>
ripple_all auto
operator*(const State<T, Dims>& l, const State<T, Dims>& r) -> State<T, Dims> {
  auto result = State<T, Dims>();
  for (size_t i = 0; i < State<T, Dims>::elements; ++i) {
    result[i] = l[i] * r[i];
  }
  return result;
}

template <typename T, size_t Dims>
ripple_all auto
operator*(const State<T, Dims>& l, const T& s) -> State<T, Dims> {
  auto result = State<T, Dims>();
  for (size_t i = 0; i < State<T, Dims>::elements; ++i) {
    result[i] = l[i] * s;
  }
  return result;
}

template <typename T, size_t Dims>
ripple_all auto
operator+(const State<T, Dims>& l, const State<T, Dims>& r) -> State<T, Dims> {
  auto result = State<T, Dims>();
  for (size_t i = 0; i < State<T, Dims>::elements; ++i) {
    result[i] = l[i] + r[i];
  }
  return result;
}

template <typename T, size_t Dims>
ripple_all auto
operator-(const State<T, Dims>& l, const State<T, Dims>& r) -> State<T, Dims> {
  auto result = State<T, Dims>();
  for (size_t i = 0; i < State<T, Dims>::elements; ++i) {
    result[i] = l[i] - r[i];
  }
  return result;
}


#endif // RIPPLE_BENCHMARK_FORCE_STATE_HPP
/**=--- ripple/benchmarks/state.hpp ------------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  state.hpp
 * \brief This file defines an implementation for a fluid state.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARK_FORCE_STATE_HPP
#define RIPPLE_BENCHMARK_FORCE_STATE_HPP

#include <ripple/container/array.hpp>
#include <ripple/math/math.hpp>
#include <ripple/storage/storage_descriptor.hpp>

/**
 * Forward declaration of state class.
 */
template <typename T, typename Dims, typename Layout>
class State;

namespace ripple {

/**
 * Specialization of the ripple ArrayTraits for the state, so that the state
 * type can inherit all array indexing and math op functionality.
 */
template <typename T, typename Dims, typename LayoutType>
struct ArrayTraits<State<T, Dims, LayoutType>> {
  // clang-format off
  /** The value type for the array. */
  using Value = std::decay_t<T>;
  /** Defines the type of the layout for the array. */
  using Layout = LayoutType;
  /** Defines the type of the array. */
  using Array  = State<Value, Dims, ripple::ContiguousOwned>;

  /** Returns the number of elements in the array. The fluid state stores
   * density and (energy/pressure), and a velocity component for each
   * dimension. */
  static constexpr size_t size = Dims::value + 2;
  // clang-format on
};

} // namespace ripple

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
template <typename T, typename Dims, typename Layout>
class State : public ripple::PolymorphicLayout<State<T, Dims, Layout>>,
              public ripple::Array<State<T, Dims, Layout>> {
  /*==--- [friends] --------------------------------------------------------==*/

  /**
   * Allows the layout traits to acess the private descriptor alias to
   * determine the layout properties of the state.
   *
   * \note This is required if the Descriptor and Storage aliases are *private*.
   */
  friend struct ripple::LayoutTraits<State, true>;

  /**
   * Allows the data from a fluid state with a different layout to be accessed
   * by this state.
   */
  template <typename TOther, typename DimsOther, typename LayoutOther>
  friend class State;

  //==--- [constants] ------------------------------------------------------==//

  // clang-format off
  /** Defines the number of dimensions for the state. */
  static constexpr size_t dims     = size_t{Dims::value};
  /** Defines the total number of elements in the state. */
  static constexpr size_t elements = dims + 2;
  /** Defines the offset to the velocity components. */
  static constexpr size_t v_offset = 2;
  /** Defines a constexpt value of 1 which is used throught the state. */
  static constexpr auto   one      = T{1};

  // clang-format off
  /** Defines the value type of the state data. */
  using ValueType   = std::remove_reference_t<T>;
  /** Defines the type of the element for the storage. */
  using ElementType = ripple::Vector<T, elements>;

  /** 
   * Defines the type for the storage descriptor for the state. The descriptor
   * needs to store two elements for the density and the pressure, and dims
   * elements for the velocity componenets.
   */
  using Descriptor = ripple::StorageDescriptor<Layout, ElementType>;
  /** Defines the type of the storage for the state. */
  using Storage    = typename Descriptor::Storage;

 public:
  /** Defines the type of the flux vector. */
  using FluxVec     = ripple::Vec<ValueType, elements, ripple::ContiguousOwned>;
  /** Defines the type of the velocity vector. */
  using VelocityVec = ripple::Vec<ValueType, dims, ripple::ContiguousOwned>;
  /** Defines the type of a contiguosu state. */
  using ContigState = State<T, Dims, ripple::ContiguousOwned>;

  // clang-format on

  /*==--- [construction] ---------------------------------------------------==*/

  /** Default constructor which creates the state. */
  ripple_all State() noexcept {}

  /**
   * Constructor to set all state elements to \p value.
   * \param value The value to set all elements to.
   */
  ripple_all State(ValueType value) noexcept {
    ripple::unrolled_for<elements>(
      [&](auto I) { storage_.template get<0, I>() = value; });
  }

  /**
   * Constructor to create the state from the storage type.
   * \param storage The storage to create this state with.
   */
  ripple_all State(Storage storage) noexcept : storage_(storage) {}

  /**
   * Constructor to create the state from another fluid type with a potentially
   * different storage layout, copying the data from the other storage to this
   * one.
   * \param  other The other fluid state to create this one from.
   * \tparam OtherLayout The storage layout of the other type.
   */
  ripple_all State(const State& other) noexcept
  : storage_{other.storage_} {}

  /**
   * Constructor to create the state from another fluid type with a potentially
   * different storage layout, copying the data from the other storage to this
   * one.
   * \param  other The other fluid state to create this one from.
   * \tparam OtherLayout The storage layout of the other type.
   */
  template <typename OtherLayout>
  ripple_all State(const State<T, Dims, OtherLayout>& other) noexcept
  : storage_{other.storage_} {}

  /**
   * Constructor to create the state from an array type (such as for fluxes).
   * \param  arr       The array to set the data from.
   * \tparam ArrayImpl The type of the array implementation.
   */
  template <typename ArrayImpl, ripple::array_enable_t<ArrayImpl> = 0>
  ripple_all State(const ArrayImpl& arr) noexcept {
    copy(arr);
  }

  /*==--- [operator overload] ----------------------------------------------==*/

  /**
   * Overload of the assignment operator to set all element of the state to
   * \p value.
   * \param value the value to set all the state data to.
   */
  ripple_all auto operator=(ValueType value) noexcept -> State& {
    ripple::unrolled_for<elements>(
      [&](auto I) { storage_.template get<0, I>() = value; });
    return *this;
  }

  /**
   * Overload of copy assignment operator to set the state from another state
   * of a potentially different layout type.
   * \param  other The other fluid state to create this one from.
   * \tparam OtherLayout The storage layout of the other type.
   */
  ripple_all auto operator=(const State& other) noexcept -> State& {
    copy(other);
    return *this;
  }

  /**
   * Overload of copy assignment operator to set the state from another state
   * of a potentially different layout type.
   * \param  other The other fluid state to create this one from.
   * \tparam OtherLayout The storage layout of the other type.
   */
  template <typename OtherLayout>
  ripple_all auto
  operator=(const State<T, Dims, OtherLayout>& other) noexcept -> State& {
    copy(other);
    return *this;
  }

  /**
   * Overload of copy assignment operator to set the state from an array type.
   * \param  arr       The array to set the data from.
   * \tparam ArrayImpl The type of the array implementation.
   */
  template <typename ArrayImpl, ripple::array_enable_t<ArrayImpl> = 0>
  ripple_all auto operator=(const ArrayImpl& arr) noexcept -> State& {
    copy(arr);
    return *this;
  }

  /**
   * Overload of operator[] to enable array functionality on the state. This
   * returns a reference to the \p ith stored element.
   * \param i The index of the element to return.
   */
  ripple_all auto operator[](size_t i) noexcept -> ValueType& {
    return storage_.template get<0>(i);
  }

  /**
   * Overload of operator[] to enable array functionality on the state. This
   * returns a constant reference to the \p ith stored element.
   * \param i The index of the element to return.
   */
  ripple_all auto
  operator[](size_t i) const noexcept -> const ValueType& {
    return storage_.template get<0>(i);
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
    return storage_.template get<0, 0>();
  }

  /**
   * Returns a const reference to the density of the fluid.
   */
  ripple_all auto rho() const noexcept -> const ValueType& {
    return storage_.template get<0, 0>();
  }

  /*==--- [velocity] -------------------------------------------------------==*/

  /**
   * Returns the density * velocity for the dimension.
   * \param  dim The index of the dimension for the component for.
   * \tparam Dim The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_all auto rho_v(Dim&& dim) const noexcept -> ValueType {
    return v(dim) * rho();
  }

  /**
   * Returns the density * velocity for the dimension.
   * \param  dim The index of the dimension for the component for.
   * \tparam Dim The type of the dimension specifier.
   */
  template <size_t Dim>
  ripple_all auto rho_v() const noexcept -> ValueType {
    return v<Dim>() * rho();
  }

  /**
   * Returns the velocity of the fluid in the \p dim dimension.
   * \param  dim The dimension to get the velocity for.
   * \tparam Dim The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_all auto v(Dim&& dim) noexcept -> ValueType& {
    return storage_.template get<0>(v_offset + static_cast<size_t>(dim));
  }

  /**
   * Returns the velocity of the fluid in the \p dim dimension.
   * \param  dim The dimension to get the velocity for.
   * \tparam Dim The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_all auto v(Dim&& dim) const noexcept -> const ValueType& {
    return storage_.template get<0>(v_offset + static_cast<size_t>(dim));
  }

  /**
   * Returns the velocity of the fluid in the Dim dimension. This overlod
   * computes the offset to the velocity component at compile time.
   * \tparam Dim The dimension to get the velocity for.
   */
  template <size_t Dim>
  ripple_all auto v() const noexcept -> const ValueType& {
    return storage_.template get<0>(v_offset + static_cast<size_t>(Dim));
  }

  /**
   * Returns the velocity of the fluid in the Dim dimension. This overlod
   * computes the offset to the velocity component at compile time.
   * \tparam Dim The dimension to get the velocity for.
   */
  template <size_t Dim>
  ripple_all auto v() noexcept -> ValueType& {
    return storage_.template get<0>(v_offset + static_cast<size_t>(Dim));
  }

  /**
   * Sets the velocity of the state in the \p dim dimension to \p value.
   * \param  dim   The dimension to set the velocity for.
   * \param  value The value to set the velocity to.
   * \tparam Dim   The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_all auto set_v(Dim&& dim, ValueType value) noexcept -> void {
    v(ripple_forward(dim)) = value;
  }

  /**
   * Sets the velocity of the state in the Dim direction to \p value. This
   * overload computes the offset to the velocity at compile time.
   * \param  value The value to set the velocity to.
   * \tparam Dim   The dimension to set the velocity in.
   */
  template <size_t Dim>
  ripple_all auto set_v(ValueType value) noexcept -> void {
    v<Dim>() = value;
  }

  /**
   * Returns a vector of the velocities.
   */
  ripple_all auto vel() noexcept -> VelocityVec {
    VelocityVec vel;
    ripple::unrolled_for<dims>([&](auto d) { vel[d] = v<d>(); });
    return vel;
  }

  /*==--- [energy] ---------------------------------------------------------==*/

  /**
   * Returns a reference to the energy of the fluid.
   */
  ripple_all auto energy() noexcept -> ValueType& {
    return storage_.template get<0, 1>();
  }

  /*
   * Returns a constant reference to the energy of the fluid.
   */
  ripple_all auto energy() const noexcept -> const ValueType& {
    return storage_.template get<0, 1>();
  }

  //==--- [pressure] -------------------------------------------------------==//

  /**
   * Returns the pressure of the fluid, using the \p eos equation of state
   * for the fluid.
   * \param  eos     The equation of state to use to compute the pressure.
   * \tparam EosImpl The implementation of the equation of state interface.
   */
  template <typename EosImpl>
  ripple_all auto
  pressure(const EosImpl& eos) const noexcept -> ValueType {
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

  /*==--- [flux] -----------------------------------------------------------==*/

  /**
   * Returns the flux for the state, in dimension \p dim.
   * \param  eos     The equation of state to use.
   * \param  dim     The dimension to get the flux in.
   * \tparam EosImpl The implementation of the eqaution of state.
   * \tparam Dim     The type of the dimension specifier.
   */
  template <typename EosImpl, typename Dim>
  ripple_all auto
  flux(const EosImpl& eos, Dim&& dim) const noexcept -> FluxVec {
    FluxVec    f;
    const auto p = pressure(eos);

    f.component(0)    = rho_v(dim);
    f.component(1)    = v(dim) * (energy() + p);
    f[v_offset + dim] = rho_v(dim) * v(dim) + p;

    // Set the remaining velocity components:
    size_t shift = 0;
    ripple::unrolled_for<dims - 1>([&](auto d) {
      if (d == dim) {
        shift++;
      }
      f[v_offset + d + shift] = rho_v(d + shift) * v(dim);
    });
    return f;
  }

  /*==--- [wavespeed] ------------------------------------------------------==*/

  /**
   * Returns the wavespeed for the state.
   * \param  eos     The equation of state for the state data.
   * \tparam EosImpl The implemenation of the equation of state interface.
   */
  template <typename EosImpl>
  ripple_all auto
  wavespeed(const EosImpl& eos) const noexcept -> ValueType {
    ValueType s = std::abs(rho_v<ripple::dimx()>());
    ripple::unrolled_for<dims - 1>([&](auto d) {
      constexpr auto dim = size_t{d} + 1;
      s                  = std::max(s, std::abs(rho_v<dim>()));
    });
    return s / rho() + eos.sound_speed(*this);
  }

 private:
  Storage storage_; //!< Storage for the state data.

  /**
   * Returns the sum of the rho_v components squared.
   */
  ripple_all auto rho_v_squared_sum() const noexcept -> ValueType {
    ValueType v_sq = ValueType{0};
    ripple::unrolled_for<dims>(
      [&](auto d) { v_sq += rho_v<d>() * rho_v<d>(); });
    return v_sq;
  }

  /**
   * Returns the sum of the velocity components squared.
   */
  ripple_all auto v_squared_sum() const noexcept -> ValueType {
    ValueType v_sq = ValueType{0};
    ripple::unrolled_for<dims>([&](auto d) { v_sq += v<d>() * v<d>(); });
    return v_sq;
  }

  /**
   * Copies the data from the \p arr array type to this state.
   * \param  arr The array type to copy data from.
   * \tparam Arr The type of the array.
   */
  template <typename Arr>
  ripple_all auto copy(const Arr& arr) noexcept -> void {
    ripple::unrolled_for<elements>([&](auto _i) {
      constexpr size_t i            = _i;
      storage_.template get<0, i>() = arr[i];
    });
  }
};

#endif // RIPPLE_BENCHMARK_FORCE_STATE_HPP
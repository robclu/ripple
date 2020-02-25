//==--- ripple/fvm/state/state_initializer.hpp ------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_initializer.hpp
/// \brief This file defines a type which can be used to initialize state data.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STATE_STATE_INITIALIZER_HPP
#define RIPPLE_STATE_STATE_INITIALIZER_HPP

#include "state_traits.hpp"
#include "state_element.hpp"
#include <ripple/core/algorithm/unrolled_for.hpp>
#include <ripple/core/container/tuple.hpp>
#include <ripple/core/math/math.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple::fv {

/// The StateInitializer type stores the values of the components of a state.
/// \tparam Ts The types of the elements to initialize the state.
template <typename... Ts>
class StateInitializer {
  /// Defines the number of elements in the initializer.
  static constexpr auto num_elements = size_t{sizeof...(Ts)};

  /// Defines the type of the container for the state elements.
  using elements_t = Tuple<Ts...>;

  elements_t _elements; //!< The elements to initialize the state with.

 public:
  /// Initializes the state initializer with the \p elements.
  ripple_host_device StateInitializer(Ts... elements)
  : _elements{elements...} {}

  /// Sets the data for the \p state. If one of the elements does not match the
  /// state type, then this will create a compile time error for the mismatch.
  /// \tparam StateImpl The implementation type of the state.
  /// \tparam EosImpl   The implementation of the equation of state.
  template <typename StateImpl, typename EosImpl>
  ripple_host_device auto set_state_data(
      State<StateImpl>&   state,
      const Eos<EosImpl>& eos 
  ) const -> void {
    using namespace ripple::math;

    constexpr auto state_elements = state_traits_t<StateImpl>::elements;
    static_assert(
      state_elements == num_elements,
      "Number of elements in initializer doesn't match number in state!"
    );

    /// First set every element to zero:
    for (auto i : range(state_traits_t<StateImpl>::elements)) {
      state[i] = 0;
    }

    // Need to make sure that the density is set first:
    unrolled_for<num_elements>([&] (auto _i) {
      constexpr auto i = size_t{_i};
      const auto& elem = get<i>(_elements);
      switch (hash(elem.name)) {
        case "rho"_hash:
          state.rho() = elem.value;
          break;
        default:
          break;
      }
    });

    // Next need to set the velocities:
    unrolled_for<num_elements>([&] (auto _i) {
      constexpr auto i = size_t{_i};
      const auto& elem = get<i>(_elements); 
      switch (hash(elem.name)) {
        case "v_x"_hash:
          state.set_v(dim_x, elem.value);
          break;
        case "v_y"_hash:
          state.set_v(dim_y, elem.value);
          break;
        case "v_z"_hash:
          state.set_v(dim_z, elem.value);
          break;
        default:
          break;
      }
    });

    // Last, set the pressure:
    unrolled_for<num_elements>([&] (auto _i) {
      constexpr auto i = size_t{_i};
      const auto& elem = get<i>(_elements);
      switch (hash(elem.name)) {
        case "pressure"_hash:
          state.set_pressure(elem.value, eos);
          break;
        default:
          break;
      }
    });
  }
};

//==--- [interface] --------------------------------------------------------==//

/// This is the interface for making a state initializer. Each of the \p
/// elements must be StateElement types.
/// \param  elements The elements to create the initializer with.
/// \tparam Es       The types of the elements.
template <typename... Es>
ripple_host_device auto make_state_initializer(Es&&... elements)
-> StateInitializer<Es...> {
  constexpr int num_state_elements = (is_state_element_v<Es> + ... + 0);

  static_assert(
    num_state_elements == sizeof...(Es),
    "All types for state initializer must be StateElement types!"
  );
  return StateInitializer<std::decay_t<Es>...>{elements...};
}

} // namespace ripple::fv

#endif // RIPPLE_STATE_STATE_INITIALZIER_HPP

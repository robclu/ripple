/**=--- ripple/benchmarks/reinitialization/element.hpp ----- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  element.hpp
 * \brief This file defines a element class for the reinitialization benchmark.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARK_LEVELSET_ELEMENT_KOKKOS_HPP
#define RIPPLE_BENCHMARK_LEVELSET_ELEMENT_KOKKOS_HPP

#include <ripple/utility/portability.hpp>

/**
 * Defines possible states for levelset reinitialization.
 */
enum class State : uint32_t { source = 0, converged = 1, updatable = 2 };

/**
 * Defines an element for re-initialization, which has a value and a state for
 * whether the element is converged.
 * \param T The type of the data for the element.
 */
template <typename T>
struct Element {
  T     value;
  State state;

  ripple_all Element() = default;

  /**
   * Constructor to set the element from another element.
   * \param e The other element to set from.
   */
  ripple_all Element(T v, State s) noexcept : value(v), state(s) {}

  /**
   * Overload of assignment operator which copies the storage.
   * \param other The other element to set this one from.
   */
  ripple_all auto operator=(const Element& other) noexcept -> Element& {
    value = other.value;
    state = other.state;
    return *this;
  }

  /**
   * Overload of assignment operato which sets the value for the element to the
   * given value.
   * \param value The value to set the element to.
   */
  ripple_all auto operator=(T val) noexcept -> Element& {
    value = val;
    return *this;
  }
};

#endif // RIPPLE_BENCHMARK_ELEMENT_KOKKOS_HPP

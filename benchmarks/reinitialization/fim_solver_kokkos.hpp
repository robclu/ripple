/**=--- ripple/benchmarks/reinitialization/fim_solver.hpp -- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  fim_solver.hpp
 * \brief This file defines an implementation of an eikonal solver using the
 *        fast iterative method.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_KOKKOS_HPP
#define RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_KOKKOS_HPP

#include "upwinder_kokkos.hpp"
#include <ripple/execution/synchronize.hpp>

template <size_t Dims>
struct DimSelector {};

/**
 * Struct to perform reinitialization using the fast iterative method.
 */
template <typename ViewType, typename T, size_t Dims>
struct FimSolver {
  static constexpr size_t dims = Dims;

  ViewType sdf;
  T        dh;
  size_t   iters;
  size_t   padding;
  size_t   elements_x;
  size_t   elements_y;

  /**
   * Constructor which sets the required state for the solver to perform
   * the solve.
   */
  FimSolver(ViewType& sdf_, T dh_, size_t it, size_t p, size_t ex, size_t ey)
  : sdf(sdf_), dh(dh_), iters(it), padding(p), elements_x(ex), elements_y(ey) {}

  /**
   * Solves the eikonal equation for for data at the given iterator.
   * \param i The index in the x dimension.
   */
  ripple_all auto operator()(int i) const -> void {
    ndim_solve_impl(i);
  }

  /**
   * Solves the eikonal equation for for data at the given iterator.
   * \param i The index in the x dimension.
   * \param j The index in the y dimension.
   */
  ripple_all auto operator()(int i, int j) const -> void {
    if (i < padding || j < padding) {
      return;
    }
    if (i > elements_x - padding || j > elements_y - padding) {
      return;
    }

    ndim_solve_impl(i, j);
  }

  /**
   * Solves the eikonal equation for for data at the given iterator.
   * \param i The index in the x dimension.
   * \param j The index in the y dimension.
   * \param j The index in the z dimension.
   */
  ripple_all auto operator()(int i, int j, int k) const -> void {
    ndim_solve_impl(i, j, k);
  }

 private:
  /**
   * Gets an element at the given index when the solver is 1D. We need these
   * overloads to have the solver implementation be N dimensional.
   */
  ripple_all auto
  get_element(DimSelector<1>, int i, int j = 0, int k = 0) const -> Element<T> {
    return sdf(i);
  }

  /**
   * Gets an element at the given index when the solver is 1D. We need these
   * overloads to have the solver implementation be N dimensional.
   */
  ripple_all auto
  get_element(DimSelector<2>, int i, int j, int k = 0) const -> Element<T> {
    return sdf(i, j);
  }

  /**
   * Gets an element at the given index when the solver is 1D. We need these
   * overloads to have the solver implementation be N dimensional.
   */
  ripple_all auto
  get_element(DimSelector<3>, int i, int j, int k) const -> Element<T> {
    return sdf(i, j, k);
  }

  /**
   * Solves for a single iteration for a cell at the given index when the
   * solver is 1D.
   */
  ripple_all auto solve(DimSelector<1>, int i, int, int) const -> T {
    constexpr T f      = T{1};
    const auto  solver = Upwinder<1>();
    return solver(sdf, dh, f, i);
  }

  /**
   * Solves for a single iteration for a cell at the given index when the
   * solver is 2D.
   */
  ripple_all auto solve(DimSelector<2>, int i, int j, int k) const -> T {
    constexpr T f      = T{1};
    const auto  solver = Upwinder<2>();
    return solver(sdf, dh, f, i, j);
  }

  /**
   * Solves for a single iteration for a cell at the given index when the
   * solver is 3D.
   */
  ripple_all auto solve(DimSelector<3>, int i, int j, int k) const -> T {
    constexpr T f      = T{1};
    const auto  solver = Upwinder<3>();
    return solver(sdf, dh, f, i, j, k);
  }

  /**
   * A generic implementation of the solve step for 1, 2, or 3 dimensions.
   */
  ripple_all void ndim_solve_impl(int i, int j = 0, int k = 0) const {
    constexpr DimSelector<dims> dim_selector = DimSelector<dims>();
    constexpr T                 tol          = T{1e-4};
    auto                        element = get_element(dim_selector, i, j, k);
    if (element.state == State::source) {
      return;
    }

    // Init the list:
    auto p = element.value, q = solve(dim_selector, i, j, k);
    if (std::abs(p) > std::abs(q)) {
      element.state = State::updatable;
    }

    int32_t iter      = 0;
    bool    updatable = false;
    while (iter++ < iters) {
      updatable = element.state == State::updatable;
      q         = solve(dim_selector, i, j, k);
      //__syncthreads();
      ripple::syncthreads();

      if (updatable) {
        p             = element.value;
        element.value = q;
        element.state = (std::abs(p - q) < tol) ? State::converged
                                                : State::updatable;
      } else {
        if (std::abs(element.value) > std::abs(q)) {
          element.value = q;
          element.state = State::updatable;
        }
      }
      //__syncthreads();
      ripple::syncthreads();

      if constexpr (dims == 3) {
        sdf(i, j, k).value = element.value;
      } else if (dims == 2) {
        sdf(i, j).value = element.value;
      }
    }
    if (element.value >= (std::numeric_limits<T>::max() - 1)) {
      element.state = State::updatable;
    }
  }
};

#endif // RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_HPP
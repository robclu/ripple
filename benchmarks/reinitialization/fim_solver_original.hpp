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

#ifndef RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_ORIGINAL_HPP
#define RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_ORIGINAL_HPP

#include "upwinder.hpp"
#include <ripple/execution/synchronize.hpp>
#include <ripple/iterator/iterator_traits.hpp>

/**
 * Struct to perform reinitialization using what is essentially the original
 * fast iterative method, but which is minimally modified for the GPU.
 */
struct FimSolverOriginal {
  /**
   * Solves the eikonal equation for for data at the given iterator.
   * \param  it       The iterator to the data to solve.
   * \param  dh       The resolution of the domain.
   * \param  iters    The number of iterations to solve for.
   * \tparam Iterator The type of the iterator.
   * \tparam T        The data type for  the resilution.
   */
  template <typename Iterator, typename T>
  ripple_all auto
  operator()(Iterator it, T dh, size_t iters) const noexcept -> void {
    static constexpr size_t dims =
      ripple::iterator_traits_t<Iterator>::dimensions;
    constexpr auto solve = Upwinder<dims>();
    constexpr auto f     = T{1};
    constexpr auto tol   = T{1e-2};

    if (it->state() == State::source) { // || it->state() != State::updatable) {
      return;
    }

    // For all cells in the list
    auto p = it->value(), q = solve(it, dh, f);
    ripple::syncthreads();
    it->value() = q;

    int32_t iter = 0;
    while (iter++ < iters) {
      // printf("%4.4f %4.4f\n", p, q);
      if (std::abs(p - q) < tol) {
        // if (p > q) {
        // printf("Iter\n");
        // Over each neighbour
        for (size_t d = 0; d < dims; ++d) {
          for (int i = -1; i <= 1; i += 2) {
            auto ngb = it.offset(d, i);
            if (ngb->state() == State::updatable) {
              continue;
            }

            // Neighbour is *not* in the list ...
            p = ngb->value();
            q = solve(ngb, dh, f);
            ripple::syncthreads();
            if (p > q) {
              ngb->value() = q;
              ngb->state() = State::updatable;
            }
          }
        }
      } else {
        it->state() = State::converged;
      }
      ripple::syncthreads();
    }
  }
};

#endif // RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_HPP
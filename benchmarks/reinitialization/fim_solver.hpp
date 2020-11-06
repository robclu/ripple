//==--- ripple/benchmarks/reinitialization/fim_solver.hpp -- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fim_solver.hpp
/// \brief This file defines an implementation of an eikonal solver using the
///        fast iterative method.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_HPP
#define RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_HPP

#include "upwinder.hpp"
#include <ripple/core/execution/synchronize.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>

/**
 * Struct to perform reinitialization using the fast iterative method.
 */
struct FimSolver {
  /**
   * Solves the eikonal equation for for data at the given iterator.
   * \param  it       The iterator to the data to solve.
   * \param  dh       The resolution of the domain.
   * \param  iters    The number of iterations to solve for.
   * \tparam Iterator The type of the iterator.
   * \tparam T        The data type for  the resilution.
   */
  template <typename Iterator, typename T>
  ripple_host_device auto
  operator()(Iterator it, T dh, size_t iters) const noexcept -> void {
    static constexpr size_t dims =
      ripple::iterator_traits_t<Iterator>::dimensions;
    constexpr auto solve = Upwinder<dims>();
    constexpr auto f     = T{1};
    constexpr auto tol   = T{1e-4};

    if (it->state() == State::source) {
      return;
    }

    // Init the list:
    auto p = it->value(), q = solve(it, dh, f);
    if (std::abs(p) > std::abs(q)) {
      it->state() = State::updatable;
    }

    int32_t iter      = 0;
    bool    updatable = false;
    while (iter++ < iters) {
      updatable = it->state() == State::updatable;
      q         = solve(it, dh, f);
      ripple::syncthreads();

      if (updatable) {
        p           = it->value();
        it->value() = q;
        it->state() = (std::abs(p - q) < tol) ? State::converged
                                              : State::updatable;
      } else {
        if (std::abs(it->value()) > std::abs(q)) {
          it->value() = q;
          it->state() = State::updatable;
        }
      }
      ripple::syncthreads();
    }
    if (it->value() >= (std::numeric_limits<T>::max() - 1)) {
      it->state() = State::updatable;
    }
  }
};

#endif // RIPPLE_BENCHMARK_REINITIALIZATION_FIM_SOLVER_HPP

//==--- ripple/benchmarks/reinitialization.cu -------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reinitialization.cu
/// \brief This file defines a benchmark for levelset reinitialization.
//
//==------------------------------------------------------------------------==//

#include "fim_solver.hpp"
#include <ripple/core/boundary/fo_extrap_loader.hpp>
#include <ripple/core/container/tensor.hpp>
#include <ripple/core/execution/executor.hpp>
#include <ripple/core/utility/timer.hpp>
#include <cuda_profiler_api.h>
#include <iostream>

/*
 * This benchmarks reinitializes levelset data using the fast iterative method.
 * Usage is:
 *    ./reinitializtion <elements per dim> <bandwidth>
 */

/** Number of dimensions for the solver. */
constexpr size_t dims = 2;

using Real    = float;
using Element = LevelsetElement<Real, ripple::StridedView>;
using Tensor  = ripple::Tensor<Element, dims>;

/**
 * Makes a tensor with the given number of elements per dimension and padding
 * elements.
 * \param elements The number of elements per dimension.
 * \param padding  The number of padding elements per side of the dimension.
 */
template <size_t Dims>
auto make_tensor(size_t elements, uint32_t padding = 0) noexcept {
  if constexpr (Dims == 1) {
    return ripple::Tensor<Element, 1>{{1}, padding, elements};
  } else if constexpr (Dims == 2) {
    return ripple::Tensor<Element, 2>{{1, 1}, padding, elements, elements};
  } else if constexpr (Dims == 3) {
    return ripple::Tensor<Element, 3>{
      {1, 1, 1}, padding, elements, elements, elements};
  }
}

int main(int argc, char** argv) {
  size_t elements = 100;
  size_t padding  = 1;
  size_t iters    = 20;
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }
  if (argc > 2) {
    iters = std::atol(argv[2]);
  }

  /*
   * NOTE: NVCC does *not* allow generic extended lambdas, so we need the type
   *       of the iterator if we want to pass lanmbdas to the methods to create
   *       the graph.
   *
   *       This is restrictive in that we need different iterators to global
   *       and shared data, and hence if we use ripple::in_shared() on the
   *       tensor data then we *also* need to change the iterator type, which is
   *       annoying.
   *
   *       We can get around this by defining the lamdas as functors with
   *       generic templates, i,e
   *
   *          template <typename It>
   *          ripple_host_device auto operator()(It&& it) const -> void {}
   *
   *       But for a simple case like this, the lamdas are nice.
   */
  auto data            = make_tensor<dims>(elements, padding);
  using Traits         = ripple::tensor_traits_t<decltype(data)>;
  using Iterator       = typename Traits::Iterator;
  using SharedIterator = typename Traits::SharedIterator;

  ripple::Graph init(ripple::ExecutionKind::gpu);
  init
    .split(
      [] ripple_host_device(Iterator it) {
        if (it.first_in_global_space()) {
          it->value() = 0;
          it->state() = State::source;
          return;
        }
        it->value() = std::numeric_limits<Real>::max();
        it->state() = State::updatable;
      },
      data)
    .then_split(
      [] ripple_host_device(Iterator it) {
        ripple::load_boundary(it, ripple::FOExtrapLoader());
      },
      data);
  ripple::execute(init);
  ripple::fence();

  ripple::Graph solve;
  Real          dh = 0.1;
  solve.split(
    [] ripple_host_device(SharedIterator it, Real dh, size_t iters) {
      constexpr auto fim_solve = FimSolver();
      fim_solve(it, dh, iters);
    },
    ripple::in_shared(data),
    dh,
    iters);

  ripple::Timer timer;
  ripple::execute(solve);
  ripple::fence();

  double elapsed = timer.elapsed_msec();
  std::cout << "Size: " << elements << "x" << elements
            << " elements, Iters: " << iters << ", Time: " << elapsed
            << " ms\n";

  if (elements < 15) {
    for (size_t j = 0; j < elements; ++j) {
      for (size_t i = 0; i < elements; ++i) {
        printf("%5.5f ", data(i, j)->value());
      }
      printf("\n");
    }
  }

  return 0;
}
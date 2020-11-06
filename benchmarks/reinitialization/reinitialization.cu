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
#include <ripple/core/graph/graph_executor.hpp>
#include <ripple/core/utility/timer.hpp>
#include <cuda_profiler_api.h>
#include <iostream>

constexpr size_t dims = 2;

using Real    = float;
using Element = LevelsetElement<Real, ripple::StridedView>;
using Tensor  = ripple::Tensor<Element, dims>;

auto make_tensor(size_t elements, uint32_t padding = 0) noexcept {
  if constexpr (dims == 1) {
    return ripple::Tensor<Element, 1>{{1}, padding, elements};
  }
  if constexpr (dims == 2) {
    return ripple::Tensor<Element, 2>{{1, 1}, padding, elements, elements};
  }
  if constexpr (dims == 3) {
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

  auto data = make_tensor(elements, padding);

  ripple::Graph init;
  init
    .split(
      [] ripple_host_device(auto&& it) {
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
      [] ripple_host_device(auto&& it) {
        ripple::load_boundary(it, ripple::FOExtrapLoader());
      },
      data);
  ripple::execute(init);
  ripple::wait_until_finished();

  ripple::Graph solve;
  Real          dh = 0.1;
  solve.split(
    [] ripple_host_device(auto&& it, Real dh, size_t iters) {
      constexpr auto fim_solve = FimSolver();
      fim_solve(it, dh, iters);
    },
    data,
    dh,
    iters);

  ripple::Timer timer;
  ripple::execute(solve);
  ripple::wait_until_finished();

  double elapsed = timer.elapsed_msec();
  std::cout << "Size: " << elements << "x" << elements
            << " elements, Iters: " << iters << ", Time: " << elapsed
            << " ms\n";

  /*
    for (int j = 0; j < elements; ++j) {
      for (int i = 0; i < elements; ++i) {
        auto v = data(i, j)->value();
        if (v > 5.0) {
          v = 5.0;
        }
        printf("%5.5f ", v);
      }
      std::cout << "\n";
    }
  */
  return 0;
}
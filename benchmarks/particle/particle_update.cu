//==--- ripple/benchmarks/position_update.cu --------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  particle_update.cu
/// \brief This file defines a benchmark for a 3D particle update.
//
//==------------------------------------------------------------------------==//

#include <iostream>
#include <ripple/core/container/tensor.hpp>
#include <ripple/core/execution/executor.hpp>
#include <ripple/core/utility/timer.hpp>
#include <cuda_profiler_api.h>
#include "particle.hpp"

/*
 * This updates the positions of 3D particles, stored in a 1D tensor.
 * Usage is:
 *  ./particle_update <num_elements>
 */

constexpr size_t dims = 3;

using Part   = Particle<Real, ripple::Num<dims>, ripple::StridedView>;
using Tensor = ripple::Tensor<Part, 1>;

int main(int argc, char** argv) {
  size_t elements = 100;
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }

  Tensor x({1}, elements);

  ripple::Graph init;
  init.split(
    [] ripple_host_device(auto&& it) {
      ripple::unrolled_for<dims>([&](auto dim) {
        it->x(dim) = it.global_idx(ripple::dim_x);
        it->v(dim) = 1.4f * static_cast<Real>(dim);
      });
    },
    x);
  ripple::execute(init);
  ripple::fence();

  const Real    dt = 0.1;
  ripple::Graph update;
  update.split(
    [] ripple_host_device(auto&& it, Real dt) { it->update(dt); }, x, dt);

  ripple::Timer timer;
  ripple::execute(update);
  ripple::fence();

  double elapsed = timer.elapsed_msec();
  std::cout << "Elements : " << elements << " : Time: " << elapsed << " ms\n";

  return 0;
}

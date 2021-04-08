/**=--- ripple/benchmarks/force.cu ------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  force.cu
 * \brief This file defines a benchmark which uses the FORCE solver.
 *
 *==------------------------------------------------------------------------==*/

#include "eos.hpp"
#include "flux.hpp"
#include "state.hpp"
#include <ripple/container/tensor.hpp>
#include <ripple/execution/executor.hpp>
#include <ripple/utility/timer.hpp>
#include <iostream>

/*
 * This computes the flux difference in each dimension using FORCE. Usage is
 *   ./force <num_elements_per_dim>
 */

/* Number of dimeneions to compute the flux difference over. */
constexpr size_t dims = 2;

using Real      = float;
using Dims      = ripple::Num<dims>;
using StateType = State<Real, Dims, ripple::StridedView>;
using Eos       = IdealGas<Real>;
using Tensor    = ripple::Tensor<StateType, dims>;

int main(int argc, char** argv) {
  size_t elements = 100;
  size_t padding  = 1;
  Real   dt = 0.1, dh = 0.2;

  if (argc > 1) {
    elements = std::atol(argv[1]);
    dt /= dh;
  }

  /* Create a tensor with (elements + padding) x (elements x padding) elements/
   * We need the padding because the flux computation uses neighbours, and we
   * want to avoid branching at the start and end of the domain. */
  Tensor x({1, 1}, padding, elements, elements);
  Eos    eos;

  // Create the graph and initialize all elements,
  // then execute and wait until finished:
  ripple::Graph init;
  init.split(
    [] ripple_all(auto&& x_iter, Eos eos) {
      x_iter->rho() = 1.0;
      x_iter->set_pressure(1.0, eos);

      // Set the velocity value in  each dimension, this is unrolled at
      // compile time:
      ripple::unrolled_for<dims>([&](auto dim) { x_iter->set_v(dim, 1.0); });
    },
    x,
    eos);
  ripple::execute(init);
  ripple::fence();

  // Create the graph to do the flux calculation:
  ripple::Graph flux;
  flux.split(
    [] ripple_all(auto&& xit, Eos eos, auto dtdh) {
      using namespace ripple;
      constexpr auto flux = Force();

      // Flux in the first dimension:
      auto f = flux(*xit, *xit.offset(dimx(), 1), eos, dimx(), dtdh) -
               flux(*xit.offset(dimx(), -1), *xit, eos, dimx(), dtdh);

      // Flux in the rest of the dimension:
      ripple::unrolled_for<dims - 1>([&](auto d) {
        constexpr auto dim = (d + 1) % dims;
        f += flux(*xit, *xit.offset(dim, 1), eos, dim, dtdh) -
             flux(*xit.offset(dim, -1), *xit, eos, dim, dtdh);
      });
      *xit = f;
    },
    x,
    eos,
    dt);

  // This will start the timer:
  ripple::Timer timer;
  ripple::execute(flux);
  ripple::barrier();

  double elapsed = timer.elapsed_msec();
  std::cout << "Size: " << elements << "x" << elements
            << " elements, Time: " << elapsed << " ms\n";

  return 0;
}
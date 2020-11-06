//==--- ripple/benchmarks/force.cu ------------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  force.cu
/// \brief This file defines a benchmark which uses the FORCE solver.
//
//==------------------------------------------------------------------------==//

#include "eos.hpp"
#include "flux.hpp"
#include "state.hpp"
#include <ripple/core/container/tensor.hpp>
#include <ripple/core/execution/executor.hpp>
#include <ripple/core/utility/timer.hpp>
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
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }

  Tensor x({1, 1}, padding, elements, elements);
  Eos    eos;

  ripple::Graph init;
  init.split(
    [] ripple_host_device(auto&& xit, Eos eos) {
      xit->rho() = 1.0;
      xit->set_pressure(1.0, eos);
      ripple::unrolled_for<dims>([&](auto dim) { xit->set_v(dim, 1.0); });
    },
    x,
    eos);
  ripple::execute(init);
  ripple::fence();

  ripple::Graph flux;
  Real          dt = 0.1, dh = 0.2;

  if (argc > 1) {
    dt /= dh;
  }
  flux.split(
    [] ripple_host_device(auto&& xit, Eos eos, auto dtdh) {
      using namespace ripple;
      constexpr auto flux = Force();

      auto f = flux(*xit, *xit.offset(dim_x, 1), eos, dim_x, dtdh) -
               flux(*xit.offset(dim_x, -1), *xit, eos, dim_x, dtdh);
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

  ripple::Timer timer;
  ripple::execute(flux);
  ripple::fence();

  double elapsed = timer.elapsed_msec();
  std::cout << "Size: " << elements << "x" << elements
            << " elements, Time: " << elapsed << " ms\n";

  return 0;
}
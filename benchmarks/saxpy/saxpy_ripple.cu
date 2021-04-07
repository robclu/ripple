/**=--- ripple/benchmarks/saxpy_ripple.cu ------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  saxpy.cu
 * \brief This file implements a saxpy bechhmark using ripple.s
 *
 *==------------------------------------------------------------------------==*/

#include "saxpy.hpp"
#include <iostream>
#include <ripple/container/tensor.hpp>
#include <ripple/execution/executor.hpp>
#include <ripple/utility/timer.hpp>
#include <iostream>

/*
 * This is a simple saxpy benchmark. Run as
 * ./saxpy_ripple <num_elements>
 */

using Tensor = ripple::Tensor<Real, 1>;

int main(int argc, char** argv) {
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }

  // Create the tensors with one partition, which will run on a single GPU.
  Tensor x({1}, elements);
  Tensor y({1}, elements);
  // ripple::executor().set_active_threads(1, 1);

  // clang-format off
  ripple::Graph init;
  init.split(
    [] ripple_host_device (auto&& xit, auto&& yit, Real xval, Real yval) {
      *xit = xval;
      *yit = yval;
  }, x, y, xval, yval);
  ripple::execute(init);
  ripple::fence();

  ripple::Graph saxpy;
  saxpy.split(
    [] ripple_host_device (auto&& xit, auto&& yit, Real a) {
      *yit = a * (*xit) + (*yit);
  }, x, y, aval);
  // clang-format on

  ripple::Timer timer;
  ripple::execute(saxpy);
  ripple::barrier();

  double elapsed = timer.elapsed_msec();
  std::cout << "Elements: " << elements << " : Time: " << elapsed << " ms\n";
  return 0;
}
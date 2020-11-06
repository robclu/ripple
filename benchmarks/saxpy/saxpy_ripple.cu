#include <iostream>
#include <ripple/core/container/tensor.hpp>
#include <ripple/core/execution/executor.hpp>
#include <ripple/core/utility/timer.hpp>
#include "saxpy.h"

/*
 * This is a simple saxpy benchmark. Run as
 * ./saxpy_ripple <num_elements>
 */

using Tensor = ripple::Tensor<Real, 1>;

int main(int argc, char** argv) {
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }

  Tensor x({1}, elements);
  Tensor y({1}, elements);
  ripple::executor().set_active_threads(1, 1);
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
  ripple::fence();

  double elapsed = timer.elapsed_msec();
  std::cout << "Elements: " << elements << " : Time: " << elapsed << " ms\n";
  return 0;
}
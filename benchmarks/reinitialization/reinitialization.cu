/**=--- ../benchmarks/reinitialization/reinitialization.cu - -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  reinitialization.cu
 * \brief This file defines a benchmark for levelset reinitialization.
 *
 *==------------------------------------------------------------------------==*/

#include "fim_solver.hpp"
#include <ripple/container/tensor.hpp>
#include <ripple/execution/executor.hpp>
#include <ripple/padding/fo_extrap_loader.hpp>
#include <ripple/utility/timer.hpp>
#include <iostream>

/*
 * This benchmarks reinitializes levelset data using the fast iterative method.
 * Usage is:
 *    ./reinitializtion <elements per dim> <bandwidth>
 */

/** Number of dimensions for the solver. */
constexpr size_t dims = 2;

using Real = float;
using Elem = Element<Real, ripple::StridedView>;

/**
 * Makes a tensor with the given number of elements per dimension and padding
 * elements.
 * \param elements The number of elements per dimension.
 * \param padding  The number of padding elements per side of the dimension.
 */
template <size_t Dims>
auto make_tensor(
  size_t elements, uint32_t padding = 0, uint32_t partitions = 1) noexcept {
  if constexpr (Dims == 1) {
    return ripple::Tensor<Elem, 1>{{1}, padding, elements};
  } else if constexpr (Dims == 2) {
    return ripple::Tensor<Elem, 2>{
      {1, partitions}, padding, elements, elements};
  } else if constexpr (Dims == 3) {
    return ripple::Tensor<Elem, 3>{
      {1, partitions, 1}, padding, elements, elements, elements};
  }
}

/**
 * Initialization functor.
 */
struct Initializer {
  /**
   * Overload of operator() to call the initializer.
   * \param  it       The iterator to initialize the data for.
   * \tparam Iterator The type of the iterator.
   */
  template <typename Iterator>
  ripple_all auto operator()(Iterator it) const noexcept -> void {
    constexpr size_t source_loc = 5;
    bool             is_source  = true;
    ripple::unrolled_for<dims>([&](auto dim) {
      if (it.global_idx(dim) != source_loc) {
        is_source = false;
      }
    });

    if (is_source) {
      it->value() = 0;
      it->state() = State::source;
    } else {
      it->value() = std::numeric_limits<Real>::max();
      it->state() = State::updatable;
    }
  }
};

int main(int argc, char** argv) {
  size_t elements   = 10;
  size_t padding    = 6;
  size_t iters      = 2;
  size_t partitions = 2;
  size_t expansion  = 2;
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }
  if (argc > 2) {
    iters = std::atol(argv[2]);
  }
  if (argc > 3) {
    expansion = std::atol(argv[3]);
  }
  if (argc > 4) {
    partitions = std::atol(argv[4]);
  }

  /*
   * NOTE: NVCC does *not* allow generic extended lambdas, so we need the type
   *       of the iterator if we want to pass lanmbdas to the methods to
   *       create the graph.
   *
   *       This is restrictive in that we need different iterators to global
   *       and shared data, and hence if we use ripple::in_shared() on the
   *       tensor data then we *also* need to change the iterator type, which
   *      is annoying.
   *
   *       We can get around this by defining the lamdas as functors with
   *       generic templates, i,e
   *
   *          template <typename It>
   *          ripple_all auto operator()(It&& it) const -> void {}
   *
   *       But for a simple case like this, the lamdas are nice.
   */
  auto data    = make_tensor<dims>(elements, padding, partitions);
  using Traits = ripple::tensor_traits_t<decltype(data)>;

  ripple::Graph init(ripple::ExecutionKind::gpu);
  /* First we initialize the data, so that the source node is set. We then
   * load the padding data for each partition, so that the values outside the
   * domain (in the padding) are valid and errors are not propogated into the
   * domain.
   */
  init.split(Initializer(), data)
    .then_split(ripple::LoadPadding(), data, ripple::FOExtrapLoader());
  ripple::execute(init);
  ripple::fence();

  ripple::Graph solve;
  Real          dh = 0.1;

  /* First we need to copy padding from the neighbour partition so that we
   * don't need to communicate during the computation, then we can execute the
   * solver, which will run for each partition. */
  solve.memcopy_padding(ripple::concurrent_padded_access(data))
    .then_split(
      FimSolver(),
      // ripple::expanded(data, expansion),
      // ripple::in_shared(data),
      data,
      ripple_move(dh),
      ripple_move(iters));

  ripple::Timer timer;
  ripple::execute(solve);
  ripple::barrier();

  double elapsed = timer.elapsed_msec();
  std::cout << "Size: " << elements << "x" << elements
            << " elements, Iters: " << iters << ", Time: " << elapsed
            << " ms\n";

  // For small sizes, print the result to test that it's working.
  if (elements < 30) {
    printf("      ");
    for (size_t i = 0; i < elements; ++i) {
      printf("%05lu ", i);
    }
    printf("\n");
    for (size_t j = 0; j < elements; ++j) {
      printf("%5lu ", j);
      for (size_t i = 0; i < elements; ++i) {
        if (data(i, j)->value() < 99.0f) {
          printf("%05.2f ", data(i, j)->value());
        } else {
          printf("----- ");
        }
      }
      printf("\n");
    }
  }

  return 0;
}
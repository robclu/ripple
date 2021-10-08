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

#include "fim_solver_kokkos.hpp"
#include <ripple/utility/timer.hpp>
#include <iostream>
#include <Kokkos_Core.hpp>

/*
 * This benchmarks reinitializes levelset data using the fast iterative method.
 * Usage is:
 *    ./reinitializtion <elements per dim> <bandwidth>
 */

/** Number of dimensions for the solver. */
constexpr size_t dims = 2;

using Real = float;
using Elem = Element<Real>;

/**
 * Makes a tensor with the given number of elements per dimension and padding
 * elements.
 * \param elements The number of elements per dimension.
 * \param padding  The number of padding elements per side of the dimension.
 */
template <size_t Dims>
auto make_view(size_t elements) noexcept {
  if constexpr (Dims == 1) {
    return Kokkos::View<Elem*>{"data", elements};
  } else if constexpr (Dims == 2) {
    return Kokkos::View<Elem**>{"data", elements, elements};
  } else if constexpr (Dims == 3) {
    return Kokkos::View<Elem***>{{"data", elements, elements, elements}};
  }
}

/**
 * Initialization functor.
 */
template <typename View>
struct Initializer {
  View   v;
  size_t padding;

  Initializer(View& view, size_t p) : v(view), padding(p) {}
  /**
   * Overload of operator() to call the initializer.
   * \param  it       The iterator to initialize the data for.
   * \tparam Iterator The type of the iterator.
   */
  ripple_all auto operator()(int i, int j) const noexcept -> void {
    const size_t source_loc = 5 + padding;
    bool         is_source  = false;
    if (i == source_loc && j == source_loc) {
      is_source = true;
    }

    auto& e = v(i, j);
    if (is_source) {
      e.value = 0;
      e.state = State::source;
    } else {
      e.value = std::numeric_limits<Real>::max();
      e.state = State::updatable;
    }
  }

  ripple_all auto operator()(int i, int j, int k) const noexcept -> void {
    constexpr size_t source_loc = 5;
    bool             is_source  = false;
    if (i == source_loc && j == source_loc && k == source_loc) {
      is_source = true;
    }

    auto& e = v(i, j, k);
    if (is_source) {
      e.value = 0;
      e.state = State::source;
    } else {
      e.value = std::numeric_limits<Real>::max();
      e.state = State::updatable;
    }
  }
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  size_t elements   = 10;
  size_t padding    = 2;
  size_t iters      = 2;
  size_t partitions = 2;
  size_t expansion  = 2;
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }
  if (argc > 2) {
    iters = std::atol(argv[2]);
  }
  if (argc > 2) {
    expansion = std::atol(argv[3]);
  }
  if (argc > 3) {
    partitions = std::atol(argv[4]);
  }

  {
    Real dh        = 0.1;
    auto data      = make_view<dims>(elements);
    using ViewType = decltype(data);

    elements += padding * 2;
    const auto range = Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
      {0, 0}, {elements, elements});
    Kokkos::parallel_for(range, Initializer<ViewType>(data, padding));
    Kokkos::fence();

    // TOOD: Copy data between partitions
    const auto solver =
      FimSolver<ViewType, Real, dims>(data, dh, iters, padding, elements);
    ripple::Timer timer;
    Kokkos::parallel_for(range, solver);
    Kokkos::fence();

    double elapsed = timer.elapsed_msec();
    std::cout << "Size: " << elements << "x" << elements
              << " elements, Iters: " << iters << ", Time: " << elapsed
              << " ms\n";

    auto m = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(m, data);
    if (elements < 30) {
      printf("      ");
      for (size_t i = 0; i < elements; ++i) {
        printf("%05lu ", i);
      }
      printf("\n");
      for (size_t j = 0; j < elements; ++j) {
        printf("%5lu ", j);
        for (size_t i = 0; i < elements; ++i) {
          if (m(i, j).value < 99.0f) {
            printf("%05.2f ", m(i, j).value);
          } else {
            printf("----- ");
          }
        }
        printf("\n");
      }
    }
  }

  Kokkos::finalize();

  return 0;
}

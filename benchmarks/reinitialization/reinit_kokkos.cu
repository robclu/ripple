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
#include <mpi.h>

/*
 * This benchmarks reinitializes levelset data using the fast iterative method.
 * Usage is:
 *    ./reinitializtion <elements per dim> <bandwidth>
 */

/** Number of dimensions for the solver. */
constexpr size_t dims = 2;

using Real       = float;
using Elem       = Element<Real>;
using BufferView = Kokkos::View<Elem*, Kokkos::HostSpace>;

/**
 * Makes a tensor with the given number of elements per dimension and padding
 * elements.
 * \param elements The number of elements per dimension.
 * \param padding  The number of padding elements per side of the dimension.
 */
template <size_t Dims>
auto make_view(size_t elements, size_t elements_y) noexcept {
  if constexpr (Dims == 1) {
    return Kokkos::View<Elem*>{"data", elements};
  } else if constexpr (Dims == 2) {
    return Kokkos::View<Elem**, Kokkos::LayoutLeft>{
      "data", elements, elements_y};
  } else if constexpr (Dims == 3) {
    return Kokkos::View<Elem***>{{"data", elements, elements_y, elements}};
  }
}

/**
 * Initialization functor to set the source data.
 * This sets a single cell to have a value of zero, and the other cells
 * to have a max value.
 */
template <typename View>
struct Initializer {
  View   v;
  size_t padding;

  Initializer(View& view, size_t p) : v(view), padding(p) {}

  /**
   * Overload of operator() to call the initializer.
   * \param i The index in the x dimension.
   * \param j The index in the y dimension.
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

  /**
   * Overload of operator() to call the initializer.
   * \param i The index in the x dimension.
   * \param j The index in the y dimension.
   * \param z The index in the z dimension.
   */
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

/*
 * This sends boundary data in two phases:
 *  - For the end of the domain, it sends from rank to rank + 1, and
 *    receives from rank-1.
 *  - For the start of the domain, it sends from rank to rank - 1, and
 *    receives from rank + 1.
 *
 * NOTE: This assumes that the data is 2D and, partitioned along the
 *       y-dimension.
 *
 *       It also assumes that only one cell of padding is used.
 *
 * \param elements The number of elements in the x-dimension.
 * \param rank     The rank of the process doing the copying.
 * \param world_size The number of processors in the world.
 */
template <typename View>
void copy_boundary_data(View& states, int elements, int rank, int world_size) {
  if (world_size == 1) {
    return;
  }
  BufferView buffer("buffer", elements);
  const int  send_elements = sizeof(Elem) * buffer.size();
  const auto mpi_type      = MPI_CHAR;

  // To send the data, we need to get a subview into the state data, and
  // then copy the subview from the device to the host, then send from the
  // host to the next rank:
  if (rank < world_size - 1) {
    auto subview = Kokkos::subview(states, Kokkos::ALL(), states.extent(1) - 1);
    auto host_subview = Kokkos::create_mirror_view(subview);
    Kokkos::deep_copy(buffer, host_subview);
    MPI_Send(
      static_cast<const void*>(buffer.data()),
      send_elements,
      mpi_type,
      rank + 1,
      42,
      MPI_COMM_WORLD);
  }
  // To receive the data we have the opposite, we create a subview of the
  // device data, and get a host representation of that, which we can receive
  // into directly.
  if (rank > 0) {
    auto       subview   = Kokkos::subview(states, Kokkos::ALL(), 0);
    auto       h_subview = Kokkos::create_mirror_view(subview);
    MPI_Status status;
    MPI_Recv(
      static_cast<void*>(h_subview.data()),
      send_elements,
      mpi_type,
      rank - 1,
      42,
      MPI_COMM_WORLD,
      &status);
    Kokkos::deep_copy(subview, h_subview);
  }

  // The same as the above processes, but from sending to a lower rank, and
  // receiving from the higher one.
  if (rank > 0) {
    auto subview      = Kokkos::subview(states, Kokkos::ALL(), 1);
    auto host_subview = Kokkos::create_mirror_view(subview);
    Kokkos::deep_copy(buffer, host_subview);
    MPI_Send(
      static_cast<const void*>(buffer.data()),
      send_elements,
      mpi_type,
      rank - 1,
      42,
      MPI_COMM_WORLD);
  }
  if (rank < world_size - 1) {
    auto subview = Kokkos::subview(states, Kokkos::ALL(), states.extent(1) - 1);
    auto h_subview = Kokkos::create_mirror_view(subview);
    MPI_Status status;
    MPI_Recv(
      static_cast<void*>(h_subview.data()),
      send_elements,
      mpi_type,
      rank + 1,
      42,
      MPI_COMM_WORLD,
      &status);
    Kokkos::deep_copy(subview, h_subview);
  }
}

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  Kokkos::initialize(argc, argv);

  size_t elements = 10;
  size_t padding  = 1;
  size_t iters    = 2;
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }
  if (argc > 2) {
    iters = std::atol(argv[2]);
  }

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t elements_per_device_y = elements / world_size;

  {
    Real dh = 0.1;
    elements += padding * 2;
    elements_per_device_y += padding * 2;
    auto data      = make_view<dims>(elements, elements_per_device_y);
    using ViewType = decltype(data);

    // Create a range object which will be used with parallel_for to specify
    // the domain of execution:
    const auto range = Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
      {0, 0}, {elements, elements_per_device_y});

    // First initialize the data:
    Kokkos::parallel_for(range, Initializer<ViewType>(data, padding));
    Kokkos::fence();

    // Next, copy the boundary data between the devices, and then run the
    // solver. For this simple benchmark, we just run a single outer loop,
    // however, to converge the whole domain, this should be placed inside
    // a loop so that data is shared between devices:
    const auto solver = FimSolver<ViewType, Real, dims>(
      data, dh, iters, padding, elements, elements_per_device_y);
    ripple::Timer timer;
    copy_boundary_data(data, elements, rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::parallel_for(range, solver);
    Kokkos::fence();

    if (rank == 0) {
      double elapsed = timer.elapsed_msec();
      std::cout << "Size: " << elements << "x" << elements
                << " elements, Iters: " << iters << ", Time: " << elapsed
                << " ms\n";

      // For debugging we can print the grid:
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
  }

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}

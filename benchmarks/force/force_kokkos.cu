/**=--- ripple/benchmarks/force/force_kokkos.cu ------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  force_kokkos.cu
 * \brief This file defines a benchmark which uses the FORCE solver.
 *
 *==------------------------------------------------------------------------==*/

#include "eos.hpp"
#include "flux_kokkos.hpp"
#include "state_kokkos.hpp"
#include <ripple/utility/timer.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>

/*
 * This computes the flux difference in each dimension using FORCE. Usage is
 *   ./force <num_elements_per_dim>
 */

/* Number of dimeneions to compute the flux difference over. */
constexpr size_t dims = 2;

using Real       = float;
using StateType  = State<Real, dims>;
using Eos        = IdealGas<Real>;
using StateView  = Kokkos::View<StateType**, Kokkos::LayoutLeft>;
using BufferView = Kokkos::View<StateType*, Kokkos::HostSpace>;

struct InitStates {
  StateView states;
  Eos       eos;

  InitStates(StateView s_, Eos e) : states(s_), eos(e) {}

  // Unfortunately, Kokkos parallel for  is only 1-dimensional, so we have
  // to work out the indices if we want a contiguous data layout =/
  KOKKOS_INLINE_FUNCTION void operator()(const int x, const int y) const {
    auto& state = states(x, y);

    state.rho() = 1.0;
    state.set_pressure(1.0, eos);

    for (size_t d = 0; d < dims; ++d) {
      state.set_v(d, 1.0);
    }
  }
};

struct UpdateStates {
  StateView states;
  Eos       eos;
  Real      dtdh;
  size_t    size_x;
  size_t    size_y;
  int       padding;

  UpdateStates(StateView s_, Eos e, Real dtdh_, size_t x, size_t y, int p_)
  : states(s_), eos(e), dtdh(dtdh_), size_x(x), size_y(y), padding(p_) {}

  // Again, we need to compute the indices into the view =/
  KOKKOS_INLINE_FUNCTION void operator()(const int x, const int y) const {
    // Check if we are a boundary element:
    if (x < padding || x >= size_x - padding - 1) {
      return;
    }
    if (y < padding || y >= size_y - padding - 1) {
      return;
    }

    constexpr auto flux = Force();

    auto f = (flux(states(x - 1, y), states(x, y), eos, 0, dtdh) -
              flux(states(x, y), states(x + 1, y), eos, 0, dtdh)) +
             (flux(states(x, y), states(x, y + 1), eos, 1, dtdh) -
              flux(states(x, y - 1), states(x, y), eos, 1, dtdh));
    states(x, y) = f;
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
 */
void copy_boundary_data(
  StateView& states, int elements, int padding, int rank, int world_size) {
  if (world_size == 1) {
    return;
  }
  BufferView     buffer("buffer", elements);
  const int      send_elements = StateType::elements * buffer.size();
  constexpr auto mpi_type      = std::is_same_v<Real, double> ? MPI_DOUBLE
                                                              : MPI_FLOAT;

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
    auto subview      = Kokkos::subview(states, Kokkos::ALL(), padding);
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

  size_t elements = 100;
  size_t padding  = 1;
  Real   dt = 0.1, dh = 0.2;

  if (argc > 1) {
    elements = std::atol(argv[1]);
    dt /= dh;
  }

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const size_t elements_per_device_y = elements / world_size;
  const size_t elements_per_device   = elements * elements_per_device_y;
  {
    Eos       eos;
    StateView states("states", elements, elements_per_device_y);
    cudaSetDevice(rank);

    // Create a range for the kernels:
    const auto range = Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
      {0, 0}, {elements, elements_per_device_y});

    Kokkos::parallel_for(range, InitStates(states, eos));
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

    ripple::Timer timer;
    copy_boundary_data(states, elements, padding, rank, world_size);

    // Perform the flux computation
    const auto state_update = UpdateStates(
      states, eos, dt, states.extent(0), states.extent(1), padding);
    Kokkos::parallel_for(range, state_update);
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      double elapsed = timer.elapsed_msec();
      std::cout << "Elements : " << elements << " : Time: " << elapsed
                << " ms\n";
    }
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
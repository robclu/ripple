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

using Real       = double;
using StateType  = State<Real, dims>;
using Eos        = IdealGas<Real>;
using StateView  = Kokkos::View<StateType**>;
using BufferView = Kokkos::View<StateType*, Kokkos::HostSpace>;

struct InitStates {
  StateView states;
  Eos       eos;

  InitStates(StateView s_, Eos e) : states(s_), eos(e) {}

  // Unfortunately, Kokkos parallel for  is only 1-dimensional, so we have
  // to work out the indices if we want a contiguous data layout =/
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const int x     = i / states.extent(0);
    const int y     = i % states.extent(0);
    auto&     state = states(x, y);

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
  int       offset_y;
  int       padding;

  UpdateStates(StateView s_, Eos e, Real dtdh_, int oy_, int p_)
  : states(s_), eos(e), dtdh(dtdh_), offset_y(oy_), padding(p_) {}

  // Again, we need to compute the indices into the view =/
  KOKKOS_INLINE_FUNCTION void operator()(const int x, const int y) const {
    //    const int x = i % states.extent(0);
    //    const int y = i / states.extent(0);

    // Check if we are a boundary element:
    if (x < padding || x >= states.extent(0) - padding - 1) {
      return;
    }
    if (y < padding || y >= states.extent(1) - padding - 1) {
      return;
    }

    constexpr auto flux = Force();

    // auto f =
    //  (flux(states(x - 1, y), states(x, y), eos, 0, dtdh) -
    //   flux(states(x, y), states(x + 1, y), eos, 0, dtdh));
    //+
    //(flux(states(x, y), states(x, y + 1), eos, 1, dtdh) -
    // flux(states(x, y - 1), states(x, y), eos, 1, dtdh));
    // states(x, y) = StateType{4};
  }
};

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
    StateView states("states", elements, elements_per_device_y);

    int offset_y = elements_per_device_y * rank;
    cudaSetDevice(rank);

    Eos        eos;
    const auto state_init = InitStates(states, eos);
    Kokkos::parallel_for(elements_per_device, state_init);
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

    ripple::Timer timer;
    // Send the boundary data between devices ...
    // Create the buffers to send data between devices:
    BufferView buffer("buffer", elements);
    const int  send_elements = StateType::elements * buffer.size();
    if (rank == 0 || rank == 1) {
      auto subview =
        Kokkos::subview(states, Kokkos::ALL(), states.extent(1) - 1);
      auto host_subview = Kokkos::create_mirror_view(subview);
      Kokkos::deep_copy(buffer, host_subview);
      MPI_Send(
        static_cast<const void*>(buffer.data()),
        send_elements,
        MPI_DOUBLE,
        rank + 1,
        42,
        MPI_COMM_WORLD);
    }
    if (rank == 1 || rank == 2) {
      auto       subview   = Kokkos::subview(states, Kokkos::ALL(), 0);
      auto       h_subview = Kokkos::create_mirror_view(subview);
      MPI_Status s;
      MPI_Recv(
        static_cast<void*>(h_subview.data()),
        send_elements,
        MPI_DOUBLE,
        rank - 1,
        42,
        MPI_COMM_WORLD,
        &s);
      Kokkos::deep_copy(subview, h_subview);
    }

    // Send the other way:
    if (rank == 1 || rank == 2) {
      auto subview      = Kokkos::subview(states, Kokkos::ALL(), padding);
      auto host_subview = Kokkos::create_mirror_view(subview);
      Kokkos::deep_copy(buffer, host_subview);
      MPI_Send(
        static_cast<const void*>(buffer.data()),
        send_elements,
        MPI_DOUBLE,
        rank - 1,
        42,
        MPI_COMM_WORLD);
    }
    if (rank == 0 || rank == 1) {
      auto subview =
        Kokkos::subview(states, Kokkos::ALL(), states.extent(1) - 1);
      auto       h_subview = Kokkos::create_mirror_view(subview);
      MPI_Status s;
      MPI_Recv(
        static_cast<void*>(h_subview.data()),
        send_elements,
        MPI_DOUBLE,
        rank + 1,
        42,
        MPI_COMM_WORLD,
        &s);
      Kokkos::deep_copy(subview, h_subview);
    }

    printf("Received second batch of data: %4lu\n ", rank);
    // Perform the flux computation
    const auto state_update = UpdateStates(states, eos, dt, offset_y, padding);
    Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
        {0, 0}, {elements, elements_per_device_y}),
      state_update);
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
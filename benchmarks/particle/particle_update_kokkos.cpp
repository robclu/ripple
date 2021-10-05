#include "particle_kokkos.hpp"
#include <ripple/utility/timer.hpp>
#include <mpi.h>

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  Kokkos::initialize(argc, argv);

  size_t elements = 100;
  if (argc > 1) {
    elements = std::atol(argv[1]);
  }

  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t elements_per_device = elements / world_size;
  {
    ViewType particles("Particles", elements);
    int      start_index = elements_per_device * rank;
    // Set the GPU to use:
    cudaSetDevice(rank);

    // Initialize the data:
    Kokkos::parallel_for(elements_per_device, InitView(particles, start_index));
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

    const Real    dt = 0.1;
    ripple::Timer timer;
    Kokkos::parallel_for(
      elements_per_device, UpdateParticles(particles, dt, start_index));
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

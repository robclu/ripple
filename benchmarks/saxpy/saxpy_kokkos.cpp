#include "saxpy.hpp"
#include <ripple/utility/timer.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

using ViewType = Kokkos::View<Real*>;

struct InitView {
  ViewType data_x;
  ViewType data_y;
  Real     x;
  Real     y;
  int      start_index;

  InitView(ViewType dx_, ViewType dy_, Real x_, Real y_, int i_)
  : data_x(dx_), data_y(dy_), x(x_), y(y_), start_index(i_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    data_x(i + start_index) = x;
    data_y(i + start_index) = y;
  }
};

struct Saxpy {
  ViewType x;
  ViewType y;
  Real     a;
  int      start_index;

  Saxpy(ViewType x_, ViewType y_, Real a_, int i_)
  : x(x_), y(y_), a(a_), start_index(i_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    y(i + start_index) = a * x(i + start_index) + y(i + start_index);
  }
};

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
    ViewType x("X", elements);
    ViewType y("Y", elements);
    int      start_index = elements_per_device * rank;
    // Set the GPU to use:
    cudaSetDevice(rank);

    // Initialize the data:
    Kokkos::parallel_for(
      elements_per_device, InitView(x, y, xval, yval, start_index));
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);

    ripple::Timer timer;
    Kokkos::parallel_for(elements_per_device, Saxpy(x, y, aval, start_index));
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
.. Ripple documentation master file, created by
   sphinx-quickstart on Tue Oct 22 14:08:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Ripple's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/utilities.rst
   api/ripple_api_root.rst

  
Ripple is a framework designed to make parallelization of large-scale 
heterogeneous applications simple, with a focus on multiple gpus systems. 
It is designed to reduce the difficulty of GPGPU programming, without 
sacrificing performance.

Currently it will use all available computational resources on a node (any
number of CPUs and/or GPUs) and is currently being extended to support 
multi-node systems.

The target compute unit is the GPU, since it offers far superior performance
compared to even large numbers of CPU cores, however, the user has a choice of
which to use, and can use both, concurrently, if desired.

Ripple uses a number of abstractions to facilitate the above, which allow for 
simple, expressive code to run in parallel across many large systems (it has 
been used to run physics simulations on grids consisting of billions of cells).

The interface for specifying computational flow is the graph interface, which
allows computational operations and dependencies to be specified expressively 
and concisely, and from which ripple can determine efficient parallelization. 
The graph interface scales well, and can scale up to 7.3x for 8 V100 GPUs for
non-trivial real-world problems

The documentation contains the API, main features, examples, and tutorials.

Building Ripple
----------------

First, get ripple from github: `ripple <https://github.com/robclu/ripple>`_

Currently, Ripple **requires** CUDA, since some of the features require it,
however, we are in the process of removing the CUDA dependency for the CPU only
use case . Ripple has the following dependencies:

.. code-block::

  cmake >= 3.19
  clang >= 9.0 with CUDA >= 9.0 or
  gcc   >= 6.0 with CUDA >= 11.0

.. note::
  Ripple is written using C++ >= 17, which is why CUDA >= 11.0 is required if
  used as the device compiler, where as clang >= 9.0 has C++-17 support and can
  be used as both the host and device compiler.

Ripple is built using cmake, specifying various options. To see all available
options for building ripple, from the project root, run

.. code-block::

  cmake -DPRINT_OPTIONS=ON .

which shows the required and optional options. 

Cmake support for CUDA does not always work as expected, so to build ripple,
the paths to the variable compilers, as well as cuda installation, need to be
specified:

.. code-block::

  mkdir build && cd build
  cmake  \
    -DCMAKE_CUDA_COMPILER=<path to cuda compiler> \
    -DCMAKE_CXX_COMPILER=<path to cxx compiler>   \
    -DCUDA_PATH=<path to cuda toolkit root>       \
    -DCMAKE_BUILD_TYPE=Release                    \
    -DCUDA_ARCHS=80;86                            \     
    ..

.. note::  
  Of the above parameters, the first three are **required**, while
  the rest are optional.

  If the cuda compiler is clang, then the CXX compiler is automatically set to
  clang as well.

  If the cuda compiler is set to nvcc, the cuda host compiler will be set to the
  :code:`CMAKE_CXX_COMPILER`.

.. note::
  If :code:`-DCMAKE_BUILD_TYPE=Debug`, the cmake language feature sometimes 
  fails to correctly verify that the cuda compiler is correct and working. The 
  current fix for this is to first specify the build parameters as above using
  :code:`-DCMAKE_BUILD_TYPE=Release`, and then simply execute 
  :code:`-DCMAKE_BUILD_TYPE=Debug ..` after.

.. note::
  Ripple will print out the complete build configuration at the end of the
  cmake command, so you can verify that the chosen parameters are correct.

Getting Started
-------------------

The shortest code to help with getting started is the SAXPY example,
which is as simple as the following:

.. code-block:: cpp

  using Tensor = ripple::Tensor<float, 1>;
  constexpr size_t size_x       = 1000000;
  constexpr size_t partitions_x = topology().num_gpus();

  // Create the tensors. Partitions is always a vector, with each component
  // specifying the number of partitions in the {x, y, z} dimension. Each
  // partition will be reside and therefore execute on a separate gpu.
  Tensor a{{partitions}, size_x};
  Tensor b{{partitions}, size_x};
  Tensor c{{partitions}, size_x};
  float x = 2.0f;

  // Create the graph, which splits the execution across the partitions of the
  // tensor. The arguments to the functors are iterators which point to the
  // cell in the tensor at the global thread indices.
  ripple::Graph graph;
  graph.split([] ripple_host_device (auto a, auto b, auto c, float x) {
    // Set a and b to thread indices:
    *a = a.global_idx(ripple::dimx());
    *b = b.global_idx(ripple::dimx());

    // Set the result:
    *c = *a * x + *b;
  }, a, b, c, x);
  ripple::execute(graph);
  ripple::fence();

  for (size_t i = 0; i < c.size(ripple::dimx()); ++i) {
    fmt::format("{} {}\n", i, *c(i));
  } 

Ripple has a lot more functionality, and the next best wat to explore some of
it is to have a look through the benchmarks, which can be found in 
:code:`benchmarks/`. If you want to build the benchmarks, then add
:code:`-DRIPPLE_BUILD_BENCHMARKS=ON` to the cmake configuration.

There is a lot more in depth information, which can be found through the 
following links:

.. toctree::
   :maxdepth: 2
   :caption: Performance:

   performance/saxpy.rst
   performance/particle_update.rst
   performance/force.rst
   performance/eikonal_solver.rst


Features
------------

There are three main components in ripple which provide all the functionality
and hence features for simple parallel programming. They all work together, and
are tensors, polymorphic data layout, and graphs. Each is given a brief
overview here, but see the additional links for more detailed information and
examples. Here we illustrate the main concepts with a simple example which
computes the dot product of a vector with itself, and then computes the finite
difference of the results. This is a contrived example, and real world examples
can be found in the :code:`benchmarks/` folder, but it illustrates the concepts
and is simple.

Tensors
^^^^^^^^^^

Tensors are and extension of arrays to multiple dimensions, and are used to
define the space on which computation is performed, they are templated over the
data type and the number of dimensions, similar to :code:`std::array`, but with
dynamic dimension sizes. The full use of tensors is only achieved when combined
with graphs, to define the operations on the tensors, and also with user-defined
classes which have polymorphic data layout. 

For our example, we will define a 2D tensor with 1000x1000 elements, with
padding element on each side of each dimension, so 1002x1002 total elements,
with a custom vector class, which we will define in the next section. We also
partition the tensor in the y dimension across all gpus.

.. code-block:: cpp

  // Alias for SoA (strided) tensor:
  using SoATensor = ripple::Tensor<Vec2<float, ripple::strided_view>, 2>;

  // To create an AoS (contiguous) tensor is as simple as:
  using AoSTensor = ripple::Tensor<Vec2<float, ripple::contiguous_view>, 2>;

  constexpr size_t size_x  = 1000;
  constexpr size_t size_y  = 1000;
  constexpr size_t padding = 1;
  std::vector partitions = {1, ripple::topology().num_gpus()};

  // Create the tensors
  SoATensor soa_x(partitions, padding, size_x, size_y);
  SoATensor soa_y(partitions, padding, size_x, size_y);

Polymorphic Data layout
^^^^^^^^^^^^^^^^^^^^^^^

For GPU codes, struct of array (SoA) data layout usually provided better
performance since it results in coalesced memory access, and therefore less
memory transactions and higher memory bandwidth. However, SoA can make software
development difficult, so ripple enables user defined classes to have 
polymorphic data layout, through a template parameters, which when used with a
tensor, will store the data as either SoA or AoS, allowing Object Oriented
classes but good performance, as well as being able to test the actual effects
on performance by changing only a few lines of code.

For our example, we will define the vector to have a polymorphic layout:

.. code-block:: cpp

  // T      : Type of the data 
  // Layout : The layout of the data
  template <typename T, typename Layout>
  struct Vec2 : ripple::PolymorphicLayout<Vec2<T, Layout>> {
    // Required by ripple, define that we want 2 elements of type T.
    using Desc    = ripple::StorageDescriptor<L, ripple::Vector<T, 2>>;
    using Storage = typename Desc::Storage;

    // Actual storage, like an array.
    Storage storage;

    // Return the x component:
    auto x() -> T& {
      // Static index syntax:
      //
      // Index of element in type ---|
      // Type index in storage ---|  |
      //                          |  |
      //                          |  |
      //                          v  v
      return storage.template get<0, 0>();
    }

    // Return the x component:
    auto y() -> T& {
      return storage.template get<0, 1>();
    }

    // Get the Ith element:
    auto operator[](size_t i) const -> T& {
      // Dynamic index syntax:
      //
      // Index of element in type --|
      // Type index in storage      |
      //                 |          |
      //                 |   |------|
      //                 v   v
      return storage.get<0>(i);
    }

    template <typename OtherLayout>
    auto dot(const Vec2<T, OtherLayout>& other) const -> T {
      return 
        storage.get<0, 0>() * other[0] + 
        storage.get<0, 1>() * other[1];
    }
  };

Graphs
^^^^^^^

Graphs are essentially the glue which bring it all together. They define the way
that the tensor data is transformed, through function objects which operate on
the tensor data, and allow the dependencies and memory transfer operations to
be specified between the operations.

All this results in a framework where a complete GPGPU program can be written 
entriely in C++ with a very minimal knowledge of GPU programming. The only real
change of mindset is that functors must be written to operate on a single 
element in the tensor, so there is no looping.

Lastly, for out example, define the graph which performs the dot product
and then the central difference.

.. note::
  Here the boundary elements (elements next to the padding cells) will be invalid because the padding cells don't compute the dot product. Ripple
  has a number of features to handle these situations, however, for simplicity,
  we don't include that here.

.. code-block:: cpp

  // Note, we could initialze the graph as
  // ripple::Graph graph(ripple::ExecutionKind::Gpu)
  // to default to gpu execution.
  ripple::Graph graph;

  // Step 1: Intialize all data to have a thread index sum:
  graph.split(
    ripple::ExecutionKind::Gpu,
    [] ripple_host_device (auto x) {
      x->x() = x.global_idx(ripple::dimx());
      x->y() = y.global_idx(ripple::dimy());
    }, soa_x);
  
  // Step 2: Set y to the dot product of x with itself:
  // This must be submitted after the previous operation, hence then_split
  graph.then_split(
    ripple::ExecutionKind::Gpu,
    [] ripple_host_device (auto x, auto y) {
    const float dot = x->dot(*x);
      y->x() = dot;
      y->y() = dot;
  }, soa_x, soa_y);

  // Step 3: Set x to the central difference using y.
  // Because there is a partition, we use concurrent data access, which will
  // perform a copy of the padding data from neighbouring partitions (i.e, the
  // neighbour dot product result computed on any adjacent cells on a different 
  // gpu:
  graph.then_split(ripple::ExecutionKind::Gpu,
    [] ripple_host_device (auto x, auto y) {
    x->x() = y.offset(ripple::dimx(), -1) + y.offset(ripple::dimx(), 1);
    x->y() = y.offset(ripple::dimy(), -1) + y.offset(ripple::dimy(), 1);
  }, soa_x, ripple::concurrent_padded_access(soa_y));

  // Graph is done, just needs execution:
  ripple::execute(graph);
  ripple::fence(); // wait until finished
   
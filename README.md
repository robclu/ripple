# Ripple

Ripple is a framework designed to make parallelization of large-scale 
heterogeneous applications simple, with a focus on multiple gpus systems. 
It is designed to reduce the difficulty of programming on systems with 
many architectures, such as is the case for modern supercomputers. The
main idea for rippe is to write once, and run on any hardware, efficiently
and in parallel, if desired.

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

Up to date documentation can be found here: [ripple docs](https://robclu.github.io/ripple_docs/)

## Building Ripple

First, get ripple from github: [ripple](https://github.com/robclu/ripple)

Currently, Ripple **requires** CUDA, since some of the features use it,
however, we are in the process of removing the CUDA dependency for the CPU only
use case . Ripple has the following dependencies:

```
cmake >= 3.19
clang >= 9.0 with CUDA >= 9.0 or
gcc   >= 6.0 with CUDA >= 11.0
```

> **NOTE:**
  Ripple is written using C++ >= 17, which is why CUDA >= 11.0 is required if
  used as the device compiler, where as clang >= 9.0 has C++-17 support and can
  be used as both the host and device compiler.

Ripple is built using cmake, specifying various options. To see all available
options for building ripple, from the project root, run

```
cmake -DPRINT_OPTIONS=ON .
```

which shows the required and optional options. 

Cmake support for CUDA does not always work as expected, so to build ripple,
the paths to the variable compilers, as well as cuda installation, need to be
specified:

```
mkdir build && cd build
cmake  \
  -DCMAKE_CUDA_COMPILER=<path to cuda compiler> \
  -DCMAKE_CXX_COMPILER=<path to cxx compiler>   \
  -DCUDA_PATH=<path to cuda toolkit root>       \
  -DCMAKE_BUILD_TYPE=Release                    \
  -DCUDA_ARCHS=80;86                            \     
  .
```

> **NOTE**
  Of the above parameters, the first three are **required**, while
  the rest are optional. If the cuda compiler is clang, then the CXX 
  compiler is automatically set to clang as well. If the cuda compiler 
  is set to nvcc, the cuda host compiler will be set to the
  `CMAKE_CXX_COMPILER`.

> **NOTE**
  If `-DCMAKE_BUILD_TYPE=Debug`, the cmake language feature sometimes 
  fails to correctly verify that the cuda compiler is correct and working. The 
  current fix for this is to first specify the build parameters as above using
  `-DCMAKE_BUILD_TYPE=Release`, and then simply execute 
  `-DCMAKE_BUILD_TYPE=Debug ..` after. 

> **NOTE**
  Ripple will print out the complete build configuration at the end of the
  cmake command, so you can verify that the chosen parameters are correct.

## Getting Started

The shortest code to help with getting started is the SAXPY example,
which is as simple as the following:

```cpp

  using Tensor = ripple::Tensor<float, 1>;
  constexpr size_t size_x     = 1000000;
  const     size_t partitions = topology().num_gpus();

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
```

Ripple has a lot more functionality, and the next best wat to explore some of
it is to have a look through the benchmarks, which can be found in 
`benchmarks/`. If you want to build the benchmarks, then add
`-DRIPPLE_BUILD_BENCHMARKS=ON` to the cmake configuration.

## Current Support

Currently, ripple will work with Intel or AMD CPUs, and NVIDIA GPUs. Over time
it is intended that ripple will support all GPUs, but that's a bit of a way off
at the moment.

# Ripple

Ripple is a library for optimized heterogeneous (CPU and GPU) compute for 
large-scale multi-dimensional spaces. It can scale to any number of CPU core
or physical GPUs on a node. In time it will likely be extended to multi-node 
systems.

The target compute unit is the GPU, since it offers far superior performance
compared to even large numbers of CPU cores, however, the user has a choice of
which to use, and can use both, concurrently, if desired.

It also includes a graph interface to specifying the structure of a computation
and the dependencies between the operations which make up the graph, from which
the library can determined the dependencies. The graph interface scales well,
and can scale up to 7.3x for 8 V100 GPUs for different example problems.

## Building Ripple

Currently, Ripple *requires* CUDA, since some of the features require it,
however, this will change shortly. Ripple has the following dependencies:

~~~
cmake >= 3.18
clang >= 9.0 with CUDA >= 9.0 or
gcc   >= 6.0 wiht CUDA >= 11.0
~~~

> **NOTE:** Ripple requires C++ >= 17, which is why CUDA >= 11.0 is required if
  used as the device compiler, where as clang >= 9.0 has C++-17 support and can
  be used as both the host and device compiler.

Ripple is built using cmake, specifying various options. To see all available
options for building ripple, from the project root, run

~~~
cmake -DPRINT_OPTIONS=ON .
~~~

which shows the required and optional options. The following shows how to build
ripple:

~~~
mkdir build && cd build
cmake  \
  -DCMAKE_CUDA_COMPILER=<path to cuda compiler> \
  -DCMAKE_CXX_COMPILER=<path to cxx compiler>   \
  -DCUDA_PATH=<path to cuda toolkit root>       \
  -DCMAKE_BUILD_TYPE=Release                    \
  -DCUDA_ARCHS=60;70                            \     
  ..
~~~

> **NOTE:** From the above parameters, the first three are *required*, while
  the rest are optional.

  If the cuda compiler is clang, then the CXX compiler is automatically set to
  clang aswell.

  If the cuda compiler is set to nvcc, the cuda host compiler will be set to the
  CMAKE_CXX_COMPILER.

## Benchmarks & Examples

To help with getting started, there are a number of annotated benchmarks and 
examples which illustrate the usage of the library. These can be found in the
`benchmarks/` and `tests/` directories, and these can be build by specifying
`-DRIPPLE_BUILD_BENCHMARKS=ON` and `-DRIPPLE_BUILD_TESTS=ON` to the build
command with cmake, respectively.


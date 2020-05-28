# Ripple

This is the C++ repository for Ripple. Ripple is essentially a framework for
optimized heterogeneous (CPU and GPU) compute on a node. In time it will be
extended to mult-node systems.

The target compute unit is the GPU, since it offers far superior performance
compared to even large numbers of CPU cores, however, the API is designed such
that all block and grid operations can be executed on the CPU or GPU.

## Overview of repository structure

### Source code

All source code is in ``RIPPLE_ROOT/include/ripple``, and there are currently
three main branches:

  - core
  - fvm
  - viz

#### Core

The core component contains general functionality, such as containers, math, 
iterators, system info, etc. It is used to implement the key functionality, 
and can be used as it's own library for heterogeneous computation.

It provides the Grid data structure, on which parallel operations can be perfomed
using conventional C++, which will execute on the CPU, or the GPU in either shared
or global memory. It allows for the data layout of struct to be changed between
AoS or SoA with a single template parameter. See the examples for details. More thorough
documentation will be available shortly.

#### FVM

The fvm component is all the finite volume related functionality. It's designed for multi-material
simulations.

#### Viz

The viz component is all visualisation and io functionality.

### Tests

The directory ``RIPPLE_ROOT/tests`` contains all the test code for the
components outlined above. They also provide examples for how the interfaces are
designed to be used.

### Examples

The directory ``RIPPLE_ROOT/examples`` contains examples and validation cases
for the overall code.

### Applications

The directory ``RIPPLE_ROOT/apps`` contains standalone applications which are
created using the functionality from the framework.

### Benchmarks

The directory ``RIPPLE_ROOT/benchmarks`` are different benchmarks cases which
test the performance of certain functionality. These are quickly put together
tests, and aren't cleaned up.

## Overview of core code design

This is a brief overview of the design of the grid and block based
infrastructure.

### Block

Blocks are the most fundametal component of Ripple. They are essentially work
units, and are designed to be run on either a single GPU or a single GPU. The
data layout of the block depends on the type stored in the block, and whether
the type is s normal user-defined type or whether it has a `StorageDescriptor`
which describes the layout. The layout can be non-owned and strided or
contiguous, or owned and contiguous. Blocks all have padding layers, which are
used for finite volume method stencils. Blocks can also be host or device, which
determines where operations on them will be performed. While blocks are the main
storage component, and the components on which compute operations are performed,
they are not the intended interface for simulations.

### Grid

The Grid is a higher level abstraction than the Block, and is made up of
multiple host and device blocks. In the case that there is only a single GPU in
the system, the grid essentially reduces to a single block. However, the
possibility of the Grid to store more blocks, and to allocate those blocks as it
likes, allows the interface to be extended to multiple GPUs and multiple nodes,
as that is required. The Grid is therefore the interface which should be used to
store data. It is multi-dimensional, and in the single-dimension case it is essentially
an std::vector which can be offloaded to the GPU.

### Operations on Grids

To perform work on grids requires a `Pipeline`. A pipeline is essentially just a
series (or a single) `Invocable` type. An invocable type is essentially just an
`std::function`, with slightly more functionality, which can be executed on
either the host or the device. A pipeline is used because it explicitly defines
synchronization between the stages of the pipeline. The design allows
for the dependencies between the stages to be determined before execution,
which removes any requirement for dynamic allocation on the GPU.

Currently all pipelines are run on the GPU blocks in the grid, however, this
will be changed once heterogeneous support is added so that the stages of the
pipeline can be run on either of the compute architectures, in parallel.


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

  
This is the official documentation for ripple. Ripple is a framework for
solving large, complex, and possibly coupled fluid and solid dynamics problems
with adaptive mesh refinement on massively parallel heterogeneous architectures.

The interface provides conceptually simple block based structures which contain
collections of cells. Iterators are provided for the blocks, and applications
can implement evolvers for the iterable data.

This documentation contains the API, as well as a lot more information. Each
section documents the rationale and intended use of the important functionality
within the module that the section documents.

Overview
-----------

Ripple is a multi-threaded heterogenous compute library. The goal of ripple is
to allow code to be easily accelerated on the GPU with CUDA, with minimal
programming effort, but with as close to optimal performance as possible.
Writing code which is fast on the GPU is difficult. There are many components
which need to be optimized. Ripple does a lot of the work for you, allowing you
to focus on the actualy implementation of the algorithm, and to optimize that.

Additionally, ripple allows for heterogeneous execution of *exactly* the same
code, so you can run using a single CPU core or 8 GPUs. Currently there is no
support for partitioning the workload across both the CPU cores and the GPUs on
a node. While this is theoretically possible, it is very difficult to get right
due to the differences in execution times of algorithms on the CPU and the GPU,
as well as the fact that the most up to date data will live in different memory
spaces. This is definitely an area with a lot of potential performance, and will
be added at a later stage.

The GPU partitions the workload into blocks which are then executed on the SMs
(streaming multiprocessors). The threads in the block execute in lock-step, and
the order of execution of the blocks is not defined. Ripple uses the same
abstraction, but makes it simpler to write efficient code. The programming model
for ripple can be broken down into two parts:

- Invocables which are applied to each grid element
- Graphs which define the invocables in an algorithm, and which are applied to
  the grid data.

The common saxpy example can be written as simply as the following:

.. code-block:: cpp

  using namespace ripple;
  Grid<double, 1> x(1000), y(1000), z(1000);

  Graph g;

  g.emplace(
    // Initialize the data to the global index in the x dimension.
    [] ripple_host_device (auto& x_it, auto& y_it) -> void {
      *x_it = global_idx(dim_x);
      *y_it = global_idx(dim_x);
    },
    // Implicit synchronization between operations:
    [] ripple_host_device (auto& x_it, auto& y_it, auto& z_it) -> void {
      const double a = 2.0;
      *z_it = a * (*x_it) + (*y_it);
    });

  g.execute(gpu_device, x, y, z);

For following is an example of the graph for the Fast Iterative method:
.. code-block:: cpp

The following sections cover the features of both.

Invocables & Blocks
--------

An invocable defines operations which are performed on an element in a Block,
which runs out of order with other blocks in a Grid. The invocable is any object
with a function call operator. It takes a multidimensional iterator to the
element in the grid. For example, and invocable which doubles every element in
the grid is as simple as:

.. code-block:: cpp

  auto double_func = [] ripple_host_device (auto& iterator) -> void {
    *it *= 2;
  }:

The following features can be used when defining invocable objects:

- Threads in a block can be synchronized with other threads in the block by
  using `sync_block()`. This essentially creates a barrier that all 
  *active* threads must reach before they continue.

- Padding can be specified for blocks when running in shared memory. This is
  useful for any application which uses stencil-type operations.

- The invocable takes a multi-dimensional iterator over the block space, which
  be offset in any dimension. Again this is useful for stencil-type operations
  which require access to local neighbours. If access is required to neighbours
  which are further apart than the padding amount, then the option to run the
  invocable in global memory can be used, which will allow any data in grid to
  be accessed from the same device.

- It can be specified which grids should be run in shared memory, which is a
  major component of good GPU performance. When specifying shared memory on the
  CPU, a thread local cache is used to mimic the GPU block, which improves cache
  hit rate.

- Blocks dimensions can be specified simply, as well as shared memory size.

- The data layout of the data can be changed between AoS and SoA with a single
  line of code, but both layout have the same OO interface of the type.
  
- [Not yet implemented] A transformation of the input grid data to a shared
  type can be specified. This is useful if the algorithm only applies to a
  subset of the data type stored in the grid, and allows better occupancy when
  running the invocables.

Portability
-----------

The header ``portability.hpp`` can be included for cross platform and cross
architecture functionality. The most important components are the macros for
marking functions as host and device. Any function, especially kernels, which is
intended to be usable on either a CPU or GPU should be marked as
``ripple_host_device``, while functions intended explicitly for execution on
the device should be marked ``ripple_device_only``. For host only functions,
the ``ripple_host_only`` macro can be used, however, by default, functions
which are not marked are host only functions. For CUDA kernels, used the marco
``ripple_global``.

Type Traits
-----------

Numerous general purpose type traits are provided which allows compile time
information to be used, overloades to be enabled and disabled, etc. Anything
which is a trait and which is general should be added here. Traits specific to
some entity, however, should be added there. For example, traits related to
Arrays should be added in the relevant files where they can be used in the
specific instances where they are appropriate. As of this writing, the latest
version of C++ which is supported on both the host and the device is c++14, for
which many of the ``_t`` and ``_v`` traits are not implement. When it is the
case that one of these traits is required, add it in the ``std::`` namespace in
the ``type_traits.hpp`` file. For example, ``std::is_same_v<T, U>`` is only
available in c++17, so to make the transition to c++17 easier, a wrapper
``std::is_same_v<T, U>`` implementation is added in the ``std::`` namespace in
the ``type_traits.hpp`` file.

Range
-----

The ``range()`` functionality is provides python-like ranges in C++. This makes
for loops cleaner and makes the resulting code more readable. For example,
instead of doing something like:

.. code-block:: cpp
  
  for (int i = 0; i < 10; ++i)
    // Do something ..

With the range functionality, the above becomes:

.. code-block:: cpp

  for (int i : range(10)):
    // Do Something ...

The ``range()`` functionality infers the type of the range index from the type
passed, and automatically starts from 0, with an increment of 1. It's possible
to specify both the start value as well as the increment, for example:

.. code-block:: cpp

  for (int i : range(2, 10, 2))
    // Range from [2 : 8]







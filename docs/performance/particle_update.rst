Particle Update Performance
======================================

We have performed a benchmark of a more complex example, which is to compute
the update of a particle in 3 dimension. For the benchmark, we compute

.. math::

  \textbf{x} = \textbf{x} + \Delta t \textbf{v}

where both x and v are vectors, and are the position and velocity of the
particle. We use a custom vector class which uses the polymorphic layout
feature from ripple so that we can test the performance of contiguous and
strided layouts.

To see the actual code implementations, see :code:`benchmarks/partice` from the 
root directory of ripple. The results presented here were run on a V100 GPU, and
we perform the benchmark with  Kokkos as well. In the table below,
(c) = contiguous layout, (s) = strided layout, (cs) = contiguous layout shared
memory, (ss) = strided layout shared memory. All times are in milliseconds.

.. list-table:: Performance comparison for particle update
   :widths: 50 50 50 50 50 50
   :header-rows: 1

   * - Size
     - Kokkos
     - Ripple (s)
     - Ripple (c)
     - Ripple (ss)
     - Ripple (cs)
   * - 100k
     - 0.0255
     - 0.0191
     - 0.0259
     - 0.0221
     - 0.0379
   * - 1M
     - 0.1444
     - 0.1036
     - 0.1620
     - 0.1343
     - 0.2381
   * - 10M
     - 1.2743
     - 0.8679
     - 1.4472
     - 1.1834
     - 2.0507
   * - 20M
     - 2.5612
     - 1.7146
     - 2.8487
     - 2.3441
     - 4.0743
   * - 40M
     - 5.0522
     - 3.4062
     - 5.7184
     - 4.6741
     - 8.1453
   * - 80M
     - 10.112
     - 6.7914
     - 11.452
     - 9.3259
     - 16.308

From the results we see that there are significant performance improvements from
using strided data, which is a result of the coalesced data access to the the
layout being SoA. 

Interestingly, the shared memory performance is **worse** for the benchmark,
essentially due to the fact that it is not necessary to use shared memory for
this since no neighbouring cell data is accessed, and the kernel is 
computationally simple.

One of the reasons that ripple has been developed is particularly for use cases 
like this, where it's simple to setup a benchmark and to determine which layout 
and which memory space is the most efficient for a particular use case.
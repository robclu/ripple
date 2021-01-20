Saxpy Performance
======================================

Ripple has been benchmarked for the simple saxpy example against other libraries
to evaluate the overhead of the implementation which facilitates the expressive
api. Specifically, we benchmark against cuBLAS and Kokkos, since cuBLAS is the
GPU library for linear algebra, and Kokkos also simplifies GPU programming.

To see the actual code implementations, see :code:`benchmarks/saxpy` from the 
root directory of ripple. The results presented here were run on a V100 GPU. 
All times are in milliseconds.

.. list-table:: Performance comparison for saxpy
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - Elements
     - cuBLAS
     - Kokkos
     - Ripple
     - Ripple NBC
   * - 1M
     - 0.0749
     - 0.0421
     - 0.0327
     - 0.0291
   * - 10M
     - 0.2316
     - 0.2722
     - 0.1704
     - 0.1119
   * - 100M
     - 1.5286
     - 2.5773
     - 1.5242
     - 1.4622
   * - 200M
     - 2.9785
     - 5.1334
     - 3.0356
     - 2.9088
   * - 500M
     - 7.3153
     - 12.826
     - 7.5261
     - 7.2347
   * - 1B
     - 14.556
     - 25.618
     - 15.048
     - 14.439

.. note::
  Ripple NBC is the performance using ripple after removing the check at the
  end of the domain to determine if the cell is inside the domain specified
  by the tensor. While this is required, we add the result to illustrate that
  the cost of the check is minimal, being a few percent for even the simplest
  kernel.

So we see here that the performance of ripple is very good, even for something 
as simple as saxpy, which is not what ripple is designed for, which is for
more complex computations across multiple gpus.
# Ripple

This is the C++ repository for Ripple. Ripple is essentially a multidimensional
block (tensor) library, with optimal data layout for parallel processing and
which can scale to very large domain sizes. 

It was initially designed for 3D fluid simulations targeted for GPU
acceleration, however, it is applicable to any application where operations can
be performed on blocks.

The target compute unit is the GPU, since it offers far superior performance
compared to even large numbers of CPU cores, however, the API is designed sich
that all block operations can be executed on the CPU or GPU.

# Inteded featues

In the longer term, the goal is to create grids composed of blocks, and then to
offload the computations on each block to either the GPU or CPU (and eventually
both, at the same time).

Additionally, the grids will be adaptive, so that then can be made coarser or
finer in certain regions.

# Upcoming features/todo list

## Features

- Add streams to blocks
- Tiling based on cache size for CPU invoke
- Threads for CPU invoke
- Multi-block array with invoke
  - This is multi-GPU, and requires that boundary data can be copied from
  - one of the blocks in the array to the others, including across devices

## Improvements

- Add pinned host memory when allocating on the device
- Add async memory allocation and copies
- Add streams to the invoke functionality for multi threaded support

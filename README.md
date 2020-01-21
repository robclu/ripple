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

## Grid algorithms

A grid comprises of blocks of different sizes, where each block has a padding
layer. The blocks in a grid can either execute on the host or the device. 

For each block, the following steps are requires to complete the execution:

1. Set the boundary/padding data for the block.
2. Run the computational kernel on the block data.
3. Update the cache state of the block (if the block is on the host/device)

Step 1 above can be it's own task, with steps 2 and 3 comprising another task,
which depends on the previous task.

Initially, break grid into segments of the number of GPUs, and have each thread
submit the work to the GPU. (Leaving all additional CPU cores idle).

Later, add a mechanism for sending the tasks to either the CPU or the GPU.


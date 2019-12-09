# Ripple

This is the C++ repository for Ripple -- a library for performing and
visualisting large-scale multiphysics applications. The library can utilise 1000s
of cores to achieve state of the art realism and performance.

# Upcoming features/todo list

- Multi-block invoke function
- Tiling based on cache size for CPU invoke
- Threads for CPU invoke
- Multi-block array with invoke
  - This is multi-GPU, and requires that boundary data can be copied from
  - one of the blocks in the array to the others, including across devices

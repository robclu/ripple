# Todo

This file is essentially a journal of the tasks which have been, and which need
to be completed, in as much detail as is required to implement the features
well.

# Complete

This section contains all completed tasks, in reverse chronological order.

# Queue

This sections contains the queue of tasks which need to be completely. They are
initially at the level of 'longer', which means to take a while to implement,
but which are then broken into weekly, daily, and hourly tasks.

## Week 23-03-2020 - 29-03-2020

- Complete Riemann Ghost Fluid mixed material implementation
    - Walk over each of the materials in the system
    - For each of the cells which are _inside_ the levelset, and _also_ 
      _adjacent_ to the boundary.
    - Then walk over all _other_ materials, and check if they are _outside_ the 
      levelset and _also_ _adjacent_ to the boundary.
    - In which case the the two starting GFM cells are found.
    - Compute the _outward_ pointing normal, and $|\phi|$ from the _inside_
      levelset.
    - For the left state, walk $|\phi| - 1.5dh$ and interpolate.
    - For the rigths tate, walk $|\phi| + 1.5dh$ and interpolate.
    - Rotate the velocities into the normal by the dot product of the velocity
      with the normal.
    - Solve for the 1D start state of states with the normal velocities.
    - Rotate the star state _back_ into the original frame.
    - Compute the left/right star state, and set the original state to it.
- Complete Fast Iterative Method for extrapolation to a bandwidth
    - The interface should be `extrapolate(mat_it, T dh, size_t bandwidth)`.
- Complete levelset velocity update implementation

## Longer term

- Implement multi-threaded, heterogenous tasking module
- Add slices to Grids
- Restructor project into _core_ and _fvm_

# Todo

This file is essentially a journal of the tasks which have been, and which need
to be completed, in as much detail as is required to implement the features
well.

# Queue

This sections contains the queue of tasks which need to be completely. They are
initially at the level of 'longer', which means to take a while to implement,
but which are then broken into weekly, daily, and hourly tasks.

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
- Update Doxygen documentation to use Markdeep

## Longer term

- Implement multi-threaded, heterogenous tasking module
- Add slices to Grids
- Restructor project into _core_ and _fvm_

# Complete

This section contains all completed tasks, in reverse chronological order.

## Week 30-03-2020 - 06-04-2020

### 01-04-2020

- Add a for each function for a parameter pack, with the signature 
  `for_each(functor, args...) -> ForEachState` which takes a functor to apply to
  each of the arguments and returns if the loop should continue or not.
- Add for each for a Tuple, as `for_each(tuple, functor, args...)`.

### 31-03-2020

- Add linear interp function in math module, as `lerp(iter, weights)`
  - Completed for 1D, 2D, and 3D
  - Iter is an N dimensional iterator, weights are N dimensional weights for the
    interpolation.
  - Interpolation is performed linearly on N-1, dimensions, until N = 1, then
    back up. 1D = lerp, 2D = bilerp, 3D = trilerp.
  - Add tests for linterp to test correctness. Tests should cover weights which
    interp the iterator and offsets from the iterator.
  - Tests should also cover cases which may cause loss of significance, such as
    for $w=0$ and $w=1$.

### 30-03-2020

- Add linear interp function in math module, as `lerp(iter, weights)`
  - Done for 1D and 2D as of today.

## Week 23-03-2020 - 29-03-2020

### 27-03-2020

- Update BlockIterator interface
  - Add the norm for the data as `it.norm(dh = 1)` to return the normal for the
    iterator from the cell being iterated from. The norm can be computed as
    $ \frac{\nabla \phi}{|\nabla \phi|}$. 
  - Add the norm for signed distance data as `it.normsd()` to return the normal
    for the iterator when the data is known to be signed distance, which is an
    optimized computation. Same as above, but $|\nabla \phi| = 1$, so it doesn't
    need to be computed. 

- Implement `math::sqrt()` in `math` module which wraps `std::sqrt` for normal
  types, and which performs elementwise sqrt for array types.

- Update BlockIterator interface
  - Add `it.grad()` to compute the gradient of the iterator.

- Fix implementation of OwnedStorage to behave like DefaultStorage
  - There is currently a problem with types which implement the StridableLayout
    interface but which specify ripple::contiguous_owned_t as the layout type.
  - The problem is that the LayoutTraits class does not choose the correct
    storage mechanism (i.e OwnedStorage), so the data is not allocated
    correctly.
  - Add an allocator to the OwnedStorage type, and change the traits in the
    LayoutTraits class to provide the correct types for classes with
    OwnedStorage.
  - _NOTE:_ Fixed this by making Stridable types which are owned use layout
    traits for non stridable types, since that's essentially what they are.

### 26-03-2020

- Update BlockIterator interface
  - Add forward difference for a given dimension as `it.forward_diff(dim)` and
    `it.forward_diff<dim>()`.
  - Add backward difference for a given dimension as `it.backward_diff(dim)` and
    `it.backward_diff<dim>()`.
  - Add central difference for a given dimension as `it.central_diff(dim)` and
    `it.central_diff<dim>()`.
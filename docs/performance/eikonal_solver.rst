Eikonal Solver Performance
======================================

We have also performed benchmarks for more complicated use cases, one of which
is the solution to the Eikonal equation, which is:

.. math::

   H(\textbf{x}, \nabla \phi) = 
    | \nabla \phi (\textbf{x}) |^2 - \frac{1}{f(\textbf{x})^2} = 0

which is essentially solving an equation which propagages information outwards
from source nodes at a speed defined by the speed function f. This benchmark
requires many iterations for the kernel, and is a good test of shared memory
performance. 
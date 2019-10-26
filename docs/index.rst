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





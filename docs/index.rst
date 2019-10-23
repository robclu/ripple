.. Streamline documentation master file, created by
   sphinx-quickstart on Tue Oct 22 14:08:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Streamline's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/streamline_api_root.rst

  
This is the official documentation for streamline. Streamline is a framework for
solving large, complex, and possibly coupled fluid and solid dynamics problems
with adaptive mesh refinement on massively parallel heterogeneous architectures.

The interface provides conceptually simple block based structures which contain
collections of cells. Iterators are provided for the blocks, and applications
can implement evolvers for the iterable data.


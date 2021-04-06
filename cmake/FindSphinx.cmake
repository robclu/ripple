#==--- ripple/cmake/FindSphinx.cmake ----------------------------------------==#
#
#                      Copyright (c) 2020 Rob Clucas
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

find_program(
  SPHINX_EXECUTABLE
  NAMES python3 -msphinx
  DOC   "Path to sphinx-build executable")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  Sphinx
  "Failed to find sphinx-build executable"
  SPHINX_EXECUTABLE)
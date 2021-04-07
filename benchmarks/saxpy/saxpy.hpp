/**=--- ripple/benchmarks/saxpy.hpp ------------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  saxpy.hpp
 * \brief This file defines common properties for the saxpy benchmarks.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARKS_SAXPY_SAXPY_HPP
#define RIPPLE_BENCHMARKS_SAXPY_SAXPY_HPP

using Real = float;

static constexpr Real xval     = 2.0;
static constexpr Real yval     = 3.0;
static inline Real    aval     = 4.0;
static inline size_t  elements = 2048;

#endif // RIPPLE_BENCHMARKS_SAXPY_SAXPY_HPP
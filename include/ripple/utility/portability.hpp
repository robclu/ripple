//==--- ripple/utility/portability.hpp --------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  portability.hpp
/// \brief This file defines portability utilities.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_PORTABILITY_HPP
#define RIPPLE_UTILITY_PORTABILITY_HPP

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

/// Definitions for host, device, and host device functions
/// if CUDA is supported.
#if defined(__CUDACC__) || defined(__clang__)
  /// Defines if GPU functionality is available.
  #define ripple_gpu_available  true
  /// Defines if CUDA functionality is available
  #define ripple_cuda_available true
  /// Defines a speficier for a host only function.
  #define ripple_host_only      __host__
  /// Defines a specifier for a device only function.
  #define ripple_device_only    __device__
  /// Defines a speficier for a host and device function.
  #define ripple_host_device    __host__ __device__
  /// Defines a specifier for a global kernel function.
  #define ripple_global         __global__

  /// Macro for thread synchronization for the device.
  #define ripple_syncthreads() __syncthreads()
#else
  /// Defines if GPU functionality is available.
  #define ripple_gpu_available  false
  /// Defines if CUDA functionality is available
  #define ripple_cuda_available false
  /// Defines a speficier for a host only function.
  #define ripple_host_only
  /// Defines a specifier for a device only function.
  #define ripple_device_only
  /// Defines a speficier for a host and device function.
  #define ripple_host_device
  /// Defines a specifier for a global kernel function.
  #define ripple_global

  /// Macro for thread synchronization for the host.
  #define ripple_syncthreads() 
#endif

#ifndef MAX_UNROLL_DEPTH
  /// Defines the max depth for compile time unrolling.
  #define ripple_max_unroll_depth 8
#else
  /// Defines the max depth for compile time unrolling.
  #define ripple_max_unroll_depth MAX_UNROLL_DEPTH
#endif

#endif // RIPPLE_UTILITY_PORTABILITY_HPP

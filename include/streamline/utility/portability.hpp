//==--- cpp/utility/portability.hpp ------------------------ -*- C++ -*- ---==//
//
//                                Streamline
//
//                      Copyright (c) 2019 Streamline.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  portability.hpp
/// \brief This file defines utilities for portability.
//
//==------------------------------------------------------------------------==//

#ifndef STREAMLINE_UTILITY_PORTABILITY_HPP
#define STREAMLINE_UTILITY_PORTABILITY_HPP

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

/// Definitions for host, device, and host device functions
/// if CUDA is supported.
#if defined(__CUDACC__) || defined(__clang__)
  /// Defines if GPU functionality is available.
  #define streamline_gpu_available  true
  /// Defines if CUDA functionality is available
  #define streamline_cuda_available true
  /// Defines a speficier for a host only function.
  #define streamline_host_only      __host__
  /// Defines a specifier for a device only function.
  #define streamline_device_only    __device__
  /// Defines a speficier for a host and device function.
  #define streamline_host_device    __host__ __device__
  /// Defines a specifier for a global kernel function.
  #define streamline_global         __global__

  /// Macro for thread synchronization for the device.
  #define streamline_syncthreads() __syncthreads()
#else
  /// Defines if GPU functionality is available.
  #define streamline_gpu_available  false
  /// Defines if CUDA functionality is available
  #define streamline_cuda_available false
  /// Defines a speficier for a host only function.
  #define streamline_host_only
  /// Defines a specifier for a device only function.
  #define streamline_device_only
  /// Defines a speficier for a host and device function.
  #define streamline_host_device
  /// Defines a specifier for a global kernel function.
  #define streamline_global

  /// Macro for thread synchronization for the host.
  #define streamline_syncthreads() 
#endif

#ifndef MAX_UNROLL_DEPTH
  /// Defines the max depth for compile time unrolling.
  #define streamline_max_unroll_depth 8
#else
  /// Defines the max depth for compile time unrolling.
  #define streamline_max_unroll_depth MAX_UNROLL_DEPTH
#endif

#endif // STREAMLINE_UTILITY_PORTABILITY_HPP

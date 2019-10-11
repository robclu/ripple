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

#include <cuda.h>
#include <cuda_runtime.h>

/// Definitions for host, device, and host device functions
/// if CUDA is supported.
#if defined(__CUDACC__) || defined(__clang__)
  #define streamline_host_only   __host__
  #define streamline_device_only __device__
  #define streamline_host_device __host__ __device__ 
  #define streamline_global      __global__

  /// Macro for thread synchronization for the device.
  #define streamline_syncthreads() __syncthreads()
#else
  #define streamline_host_only
  #define streamline_device_only
  #define streamline_host_device
  #define streamline_global

  /// Macro for thread synchronization for the host.
  #define streamline_syncthreads() 
#endif

#endif // STREAMLINE_UTILITY_PORTABILITY_HPP

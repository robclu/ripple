//==--- ripple/core/utility/portability.hpp --------------------- -*- C++ -*-
//---==//
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
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <cooperative_groups.h>

// clang-format off
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
// clang-format on

//==--- [compiler] ---------------------------------------------------------==//

/// Defines if clang is being used.
#define ripple_clang __clang__
/// Defines if gcc is being used.
#define riplple_gcc __GNUC__ && !(__clang__) && !(__CUDACC__)
/// Defines if nvcc is being used.
#define ripple_nvcc __CUDACC__ && !(__clang__)

#ifndef MAX_UNROLL_DEPTH
  /// Defines the max depth for compile time unrolling.
  #define ripple_max_unroll_depth 8
#else
  /// Defines the max depth for compile time unrolling.
  #define ripple_max_unroll_depth MAX_UNROLL_DEPTH
#endif

namespace ripple {
namespace cuda {
namespace debug {

/// Checks if a cuda error code was a success, and if not, prints the error
/// message.
/// \param[in] err_code The cuda error code.
/// \param[in] file     The file where the error was detected.
/// \param[in] line     The line in the file where the error was detected.
inline auto
check_cuda_error(cudaError_t err_code, const char* file, int line) -> void {
  if (err_code != cudaSuccess) {
    printf(
      "\nCuda Error : %s\nFile       : %s\nLine       :  %i\n\n",
      cudaGetErrorString(err_code),
      file,
      line);
    std::terminate();
  }
}

} // namespace debug
} // namespace cuda
} // namespace ripple

#if defined(NDEBUG)
  /// Defines a macro for checking a cuda error in release mode. This does not
  /// do anything so that there is no performance cost in release mode.
  #define ripple_check_cuda_result(result) (result)
#else
  /// Defines a macro to check the result of cuda calls in debug mode.
  #define ripple_check_cuda_result(result) \
    ::ripple::cuda::debug::check_cuda_error((result), __FILE__, __LINE__)
#endif // NDEBUG

#endif // RIPPLE_UTILITY_PORTABILITY_HPP

//==--- ripple/core/utility/portability.hpp ---------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <exception>

/**
 * Clang currently has a problem including the crt device function headers for
 * sm80 and sm 86, so they have to be manually included for now.
 */
#if defined(__clang__) && defined(RIPPLE_SM_80)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-W#warnings"
  #pragma clang diagnostic ignored "-Wpedantic"
  #include <crt/sm_80_rt.h>
  #pragma clang diagnostic pop
#endif

// clang-format off
/**
 * Definitions for host, device, and host device functions
 * if CUDA is supported by either nvcc or clang.
 */
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__)) 
  /** Defines if GPU functionality is available. */
  #define ripple_gpu_available  true
  /** Defines if CUDA functionality is available */
  #define ripple_cuda_available true
  /** Defines a speficier for a host only function. */
  #define ripple_host           __host__
  /** Defines a specifier for a device only function. */
  #define ripple_device         __device__
  /** Defines a speficier for a host and device function. */
  #define ripple_host_device    __host__ __device__
  /** Defines a specifier for a global kernel function. */
  #define ripple_global         __global__
  /* Defines valid code if cuda is available. */
  #define ripple_if_cuda(...)   __VA_ARGS__

  #if defined(__CUDA_ARCH__)
    /** Compiling for the gpu */
    #define ripple_gpu_compile
  #else
    /** Compiling for the cpu. */
    #define ripple_cpu_compile
  #endif 
#else
  /** Defines if GPU functionality is available. */
  #define ripple_gpu_available  false
  /** Defines if CUDA functionality is available */
  #define ripple_cuda_available false
  /** Defines a speficier for a host only function. */
  #define ripple_host
  /** Defines a specifier for a device only function. */
  #define ripple_device
  /** Defines a speficier for a host and device function. */
  #define ripple_host_device
  /** Defines a specifier for a global kernel function. */
  #define ripple_global
  /** Removes the code defined in the argument. */
  #define ripple_if_cuda(...) 
  /** Compiling for the cpu. */
  #define ripple_cpu_compile
#endif
// clang-format on

/*==--- [compiler] ---------------------------------------------------------==*/

/** Defines if clang is being used. */
#define ripple_clang __clang__
/** Defines if gcc is being used. */
#define riplple_gcc __GNUC__ && !(__clang__) && !(__CUDACC__)
/** Defines if nvcc is being used. */
#define ripple_nvcc __CUDACC__ && !(__clang__)

#ifndef RIPPLE_MAX_UNROLL_DEPTH
  /** Defines the max depth for compile time unrolling. */
  #define ripple_max_unroll_depth 8
#else
  /** Defines the max depth for compile time unrolling. */
  #define ripple_max_unroll_depth RIPPLE_MAX_UNROLL_DEPTH
#endif

#if __cplusplus >= 201703L
  /** Macro for nodiscard */
  #define ripple_nodiscard [[nodiscard]]
#else
  /** Macro for nodiscard */
  #define ripple_nodiscard
#endif

namespace ripple {

/** Defines an alias for a cuda stream based on cuda availability. */
using GpuStream =
#if defined(ripple_cuda_available)
  cudaStream_t;
#else
  int;
#endif

/** Defines an alias for a gpu error based on cuda availability. */
using GpuError =
#if defined(ripple_cuda_available)
  cudaError_t;
#else
  int;
#endif

/**
 * Defines an alias for a gpu event.
 */
using GpuEvent =
#if defined(ripple_cuda_available)
  cudaEvent_t;
#else
  int;
#endif

/** Defines the default gpu stream. */
static constexpr GpuStream default_gpu_stream = 0;

} // namespace ripple

namespace ripple::gpu::debug {

/**
 * Checks if a cuda error code was a success, and if not, prints the error
 * message.
 * \param err_code The cuda error code.
 * \param file     The file where the error was detected.
 * \param line     The line in the file where the error was detected.
 */
inline auto
check_cuda_error(GpuError err_code, const char* file, int line) -> void {
#if defined(ripple_cuda_available)
  if (err_code != cudaSuccess) {
    printf(
      "\nCuda Error : %s\nFile       : %s\nLine       :  %i\n\n",
      cudaGetErrorString(err_code),
      file,
      line);
    std::terminate();
  }
#endif
}

} // namespace ripple::gpu::debug

#if defined(NDEBUG)
  /**
   * Defines a macro for checking a cuda error in release mode. This does not
   * do anything so that there is no performance cost in release mode.
   */
  #define ripple_check_cuda_result(result) ripple_if_cuda(result)
#else
  /** Defines a macro to check the result of cuda calls in debug mode. */
  #define ripple_check_cuda_result(result) \
    ripple_if_cuda(                        \
      ::ripple::gpu::debug::check_cuda_error((result), __FILE__, __LINE__))
#endif // NDEBUG

#endif // RIPPLE_UTILITY_PORTABILITY_HPP
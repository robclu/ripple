//==--- ripple/utility/cuda.hpp ---------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cuda.hpp
/// \brief This file defines utilities for cuda.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_CUDA_HPP
#define RIPPLE_UTILITY_CUDA_HPP

#include "detail/copy_impl_.cuh"
#include "portability.hpp"

namespace ripple {
namespace cuda   {

/// Copies \p bytes of data from \p dev_ptr to \p dev_ptr.
/// \param[in]  dev_ptr_in   The device pointer to copy from.
/// \param[in]  dev_ptr_out  The device pointer to copy to.
/// \param[in]  bytes        The number of bytes to copy.
/// \tparam     DevPtr       The type of the device pointer.
template <typename DevPtr>
static inline auto memcpy_device_to_device(
  DevPtr* dev_ptr_out, const DevPtr* dev_ptr_in,std::size_t bytes
) -> void {
#if defined(__CUDACC__)
  constexpr auto num_threads = 2048;
  const     auto elements    = bytes / sizeof(DevPtr);
  auto threads = dim3(num_threads);
  auto blocks  = dim3(elements / num_threads);

  kernel::copy<<<blocks, threads>>>(dev_ptr_out, dev_ptr_in, elements);
  ripple_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__
}

/// Copies \p bytes of data from \p host_ptr to \p dev_ptr.
/// \param  dev_ptr  The device pointer to copy to.
/// \param  host_ptr The host pointer to copy from.
/// \param  bytes    The number of bytes to copy.
/// \tparam DevPtr   The type of the device pointer.
/// \tparam HostPtr  The type of the host pointer.
template <typename DevPtr, typename HostPtr>
static inline auto memcpy_host_to_device(
  DevPtr* dev_ptr, const HostPtr* host_ptr, std::size_t bytes
) -> void {
  ripple_check_cuda_result(
    cudaMemcpy(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice)
  );
}

/// Copies \p bytes of data from \p dev_ptr to \p host_ptr.
/// \param  host_ptr The host pointer to copy from.
/// \param  dev_ptr  The device pointer to copy to.
/// \param  bytes    The number of bytes to copy.
/// \tparam HostPtr  The type of the host pointer.
/// \tparam DevPtr   The type of the device pointer.
template <typename HostPtr, typename DevPtr>
static inline auto memcpy_device_to_host(
  HostPtr* host_ptr, const DevPtr* dev_ptr, std::size_t bytes
) -> void {
  ripple_check_cuda_result(
    cudaMemcpy(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost)
  );
}

/// Allocates \p bytes of memory on the device pointer pointed to by \p dev_ptr.
/// \param[in] dev_ptr The device pointer to allocate memory for.
/// \param[in] bytes   The number of bytes to allocate.
/// \tparam    Ptr     The type of the pointer.
template <typename Ptr>
static inline auto allocate(Ptr** dev_ptr, std::size_t bytes) -> void {
  ripple_check_cuda_result(cudaMalloc((void**)dev_ptr, bytes));
}

/// Frees the pointer \p ptr.
/// \param[in] ptr The pointer to free.
/// \tparam    Ptr The type of the pointer to free.
template <typename Ptr>
static inline auto free(Ptr* ptr) -> void {
  ripple_check_cuda_result(cudaFree(ptr));
}

}} // namespace ripple::cuda

#endif // RIPPLE_UTILITY_CUDA_HPP

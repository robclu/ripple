//==--- ripple/core/utility/cuda.hpp ---------------------------- -*- C++ -*- ---==//
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

//==--- [device to device]--------------------------------------------------==//

/// Copies \p bytes of data from \p dev_ptr to \p dev_ptr.
/// This will block on the host until the copy is complete.
/// \param[in]  dev_ptr_in   The device pointer to copy from.
/// \param[in]  dev_ptr_out  The device pointer to copy to.
/// \param[in]  bytes        The number of bytes to copy.
/// \tparam     DevPtr       The type of the device pointer.
template <typename DevPtr>
static inline auto memcpy_device_to_device(
  DevPtr* dev_ptr_out, const DevPtr* dev_ptr_in, std::size_t bytes
) -> void {
  ripple_check_cuda_result(
    cudaMemcpy(dev_ptr_out, dev_ptr_in, bytes, cudaMemcpyDeviceToDevice)
  );
}

/// Copies \p bytes of data from \p dev_ptr to \p dev_ptr asynchronously.
/// This will not block on the host, and will likely return before the copy is
/// complete.
/// \param[in]  dev_ptr_in   The device pointer to copy from.
/// \param[in]  dev_ptr_out  The device pointer to copy to.
/// \param[in]  bytes        The number of bytes to copy.
/// \tparam     DevPtr       The type of the device pointer.
template <typename DevPtr>
static inline auto memcpy_device_to_device_async(
  DevPtr* dev_ptr_out, const DevPtr* dev_ptr_in, std::size_t bytes
) -> void {
  ripple_check_cuda_result(
    cudaMemcpyAsync(dev_ptr_out, dev_ptr_in, bytes, cudaMemcpyDeviceToDevice)
  );
}

/// Copies \p bytes of data from \p dev_ptr to \p dev_ptr asynchronously.
/// This will not block on the host, and will likely return before the copy is
/// complete. The copy operation is scheduled on the \p stream, which will allow
/// it to be overlapped with operations on other streams, but wil not overlap
/// with computation on the device.
///
/// \param[in]  dev_ptr_in   The device pointer to copy from.
/// \param[in]  dev_ptr_out  The device pointer to copy to.
/// \param[in]  bytes        The number of bytes to copy.
/// \param[in]  stream       The stream to perform the copy on.
/// \tparam     DevPtr       The type of the device pointer.
template <typename DevPtr>
static inline auto memcpy_device_to_device_async(
  DevPtr*             dev_ptr_out, 
  const DevPtr*       dev_ptr_in ,
  std::size_t         bytes      ,
  const cudaStream_t& stream
) -> void {
  ripple_check_cuda_result(
    cudaMemcpyAsync(
      dev_ptr_out             ,
      dev_ptr_in              ,
      bytes                   ,
      cudaMemcpyDeviceToDevice,
      stream
    )
  );
}

//==--- [host to device] ---------------------------------------------------==//

/// Copies \p bytes of data from \p host_ptr to \p dev_ptr. This will block
/// until the copy completes.
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

/// Copies \p bytes of data from \p host_ptr to \p dev_ptr, asynchronously. 
/// This will not block, and will likley return before the copy is complete.
/// This requires that the \p host_ptr data be page locked, which should have
/// been allocated with alloc_pinned().
///
/// \param  dev_ptr  The device pointer to copy to.
/// \param  host_ptr The host pointer to copy from.
/// \param  bytes    The number of bytes to copy.
/// \tparam DevPtr   The type of the device pointer.
/// \tparam HostPtr  The type of the host pointer.
template <typename DevPtr, typename HostPtr>
static inline auto memcpy_host_to_device_async(
  DevPtr* dev_ptr, const HostPtr* host_ptr, std::size_t bytes
) -> void {
  ripple_check_cuda_result(
    cudaMemcpyAsync(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice)
  );
}

/// Copies \p bytes of data from \p host_ptr to \p dev_ptr, asynchronously. 
/// This will not block, and will likley return before the copy is complete.
/// This requires that the \p host_ptr data be page locked, which should have
/// been allocated with alloc_pinned(). This overload alows the stream to be
/// specified, which will allow the copy operation to be overlapped with
/// operations in other streams, and possibly computation in the same stream if
/// suppored.
///
/// \param  dev_ptr  The device pointer to copy to.
/// \param  host_ptr The host pointer to copy from.
/// \param  bytes    The number of bytes to copy.
/// \param  stream   The stream to assign the copy to.
/// \tparam DevPtr   The type of the device pointer.
/// \tparam HostPtr  The type of the host pointer.
template <typename DevPtr, typename HostPtr>
static inline auto memcpy_host_to_device_async(
  DevPtr*             dev_ptr ,
  const HostPtr*      host_ptr,
  std::size_t         bytes   ,
  const cudaStream_t& stream
) -> void {
  ripple_check_cuda_result(
    cudaMemcpyAsync(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, stream)
  );
}

//==--- [device to host] ---------------------------------------------------==//

/// Copies \p bytes of data from \p dev_ptr to \p host_ptr.
/// This will block on the host until the copy is complete.
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

/// Copies \p bytes of data from \p dev_ptr to \p host_ptr, asynchronously. This
/// will not block on the host, and will likely return before the copy is
/// complete.
/// \param  host_ptr The host pointer to copy from.
/// \param  dev_ptr  The device pointer to copy to.
/// \param  bytes    The number of bytes to copy.
/// \tparam HostPtr  The type of the host pointer.
/// \tparam DevPtr   The type of the device pointer.
template <typename HostPtr, typename DevPtr>
static inline auto memcpy_device_to_host_async(
  HostPtr* host_ptr, const DevPtr* dev_ptr, std::size_t bytes
) -> void {
  ripple_check_cuda_result(
    cudaMemcpyAsync(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost)
  );
}

/// Copies \p bytes of data from \p dev_ptr to \p host_ptr, asynchronously, on
/// the \p stream. This will not block on the host, and will likely return before 
/// the copy is complete. Specifying the \p stream allows the copy to be
/// overlapped with operations in other streams, and potentially with
/// computation in the \p stream, if supported by the device.
///
/// \param  host_ptr The host pointer to copy from.
/// \param  dev_ptr  The device pointer to copy to.
/// \param  bytes    The number of bytes to copy.
/// \param  stream   The stream to perform the copy on.
/// \tparam HostPtr  The type of the host pointer.
/// \tparam DevPtr   The type of the device pointer.
template <typename HostPtr, typename DevPtr>
static inline auto memcpy_device_to_host_async(
  HostPtr*            host_ptr,
  const DevPtr*       dev_ptr ,
  std::size_t         bytes   ,
  const cudaStream_t& stream
) -> void {
  ripple_check_cuda_result(
    cudaMemcpyAsync(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost, stream)
  );
}

//==--- [allocation device] ------------------------------------------------==//

/// Allocates \p bytes of memory on the device pointer pointed to by \p dev_ptr.
/// \param[in] dev_ptr The device pointer to allocate memory for.
/// \param[in] bytes   The number of bytes to allocate.
/// \tparam    Ptr     The type of the pointer.
template <typename Ptr>
static inline auto allocate_device(Ptr** dev_ptr, std::size_t bytes) -> void {
  ripple_check_cuda_result(cudaMalloc((void**)dev_ptr, bytes));
}

/// Frees the pointer \p ptr.
/// \param[in] ptr The pointer to free.
/// \tparam    Ptr The type of the pointer to free.
template <typename Ptr>
static inline auto free_device(Ptr* ptr) -> void {
  ripple_check_cuda_result(cudaFree(ptr));
}

//==--- [allocation host] --------------------------------------------------==//

/// Allocates \p bytes of page locked (pinned) memory on the host pointer
/// pointed to by \p host_ptr.
/// \param  host_ptr  The host pointer for the allocated memory.
/// \param  bytes     The number of bytes to allocate.
/// \tparam Ptr       The type of the pointer.
template <typename Ptr>
static inline auto allocate_host_pinned(Ptr** host_ptr, std::size_t bytes) 
-> void {
  ripple_check_cuda_result(
    cudaHostAlloc((void**)host_ptr, bytes, cudaHostAllocPortable)
  );
}

/// Frees the pointer \p ptr which was allocated as pinned memory..
/// \param[in] ptr The pointer to free.
/// \tparam    Ptr The type of the pointer to free.
template <typename Ptr>
static inline auto free_host_pinned(Ptr* ptr) -> void {
  ripple_check_cuda_result(cudaFreeHost(ptr));
}

}} // namespace ripple::cuda

#endif // RIPPLE_UTILITY_CUDA_HPP

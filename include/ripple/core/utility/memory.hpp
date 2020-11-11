//==--- ripple/core/utility/memory.hpp --------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  memory.hpp
/// \brief This file defines a utility functions for memory related operations.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_UTILITY_MEMORY_HPP
#define RIPPLE_UTILITY_MEMORY_HPP

#include "portability.hpp"
#include <cassert>
#include <cstdint>

namespace ripple {

/**
 * Gets a new ptr offset by the given amount from the ptr.
 *
 * \note This does __not__ ensure alignemt. If the pointer needs to be aligned,
 *       then pass the result to `align()`.
 *
 * \sa align
 *
 * \param ptr    The pointer to offset.
 * \param amount The amount to offset ptr by.
 * \return A new pointer at the offset location.
 */
ripple_host_device static inline auto
offset_ptr(const void* ptr, uint32_t amount) noexcept -> void* {
  return reinterpret_cast<void*>(uintptr_t(ptr) + amount);
}

/**
 * Gets a pointer with an address aligned to the goven alignment.
 *
 * \note In debug, this will assert at runtime if the alignemnt is not a power
 *       of two, in release, the behaviour is undefined.
 *
 * \param ptr       The pointer to align.
 * \param alignment The alignment to ensure.
 */
ripple_host_device static inline auto
align_ptr(const void* ptr, size_t alignment) noexcept -> void* {
  assert(
    !(alignment & (alignment - 1)) &&
    "Alignment must be a power of two for linear allocation!");
  return reinterpret_cast<void*>(
    (uintptr_t(ptr) + alignment - 1) & ~(alignment - 1));
}

namespace gpu {

/*==--- [device to device]--------------------------------------------------==*/

/**
 * Copies the given bytes of data from one device pointer to the other.
 *
 * \note This will block on the host until the copy is complete.
 *
 * \param  dev_ptr_in   The device pointer to copy from.
 * \param  dev_ptr_out  The device pointer to copy to.
 * \param  bytes        The number of bytes to copy.
 * \tparam DevPtr       The type of the device pointer.
 */
template <typename DevPtr>
static inline auto memcpy_device_to_device(
  DevPtr* dev_ptr_out, const DevPtr* dev_ptr_in, size_t bytes) -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpy(dev_ptr_out, dev_ptr_in, bytes, cudaMemcpyDeviceToDevice)));
}

/**
 * Copies given number of bytes of data from the one device pointer to the
 * other device ptr, asynchronously.
 *
 * \note This will not block on the host, and will likely return before the
 *       copy is complete.
 *
 * \param  dev_ptr_in   The device pointer to copy from.
 * \param  dev_ptr_out  The device pointer to copy to.
 * \param  bytes        The number of bytes to copy.
 * \tparam DevPtr       The type of the device pointer.
 */
template <typename DevPtr>
static inline auto memcpy_device_to_device_async(
  DevPtr* dev_ptr_out, const DevPtr* dev_ptr_in, size_t bytes) -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpyAsync(dev_ptr_out, dev_ptr_in, bytes, cudaMemcpyDeviceToDevice)));
}

/**
 * Copies given number of bytes of data from the one device pointer to the
 * other device ptr, asynchronously on the given stream.
 *
 * \note This will not block on the host, and will likely return before the
 *       copy is complete.
 *
 * \param  dev_ptr_in   The device pointer to copy from.
 * \param  dev_ptr_out  The device pointer to copy to.
 * \param  bytes        The number of bytes to copy.
 * \param  stream       The stream to perform the copy on.
 * \tparam DevPtr       The type of the device pointer.
 */
template <typename DevPtr>
static inline auto memcpy_device_to_device_async(
  DevPtr* dev_ptr_out, const DevPtr* dev_ptr_in, size_t bytes, GpuStream stream)
  -> void {
  ripple_check_cuda_result(ripple_if_cuda(cudaMemcpyAsync(
    dev_ptr_out, dev_ptr_in, bytes, cudaMemcpyDeviceToDevice, stream)));
}

/*==--- [host to device] ---------------------------------------------------==*/

/**
 * Copies the given number of bytes of data from the host pointer to the device
 * pointer.
 *
 * \note This will block on the host until the copy completes.
 *
 * \param  dev_ptr  The device pointer to copy to.
 * \param  host_ptr The host pointer to copy from.
 * \param  bytes    The number of bytes to copy.
 * \tparam DevPtr   The type of the device pointer.
 * \tparam HostPtr  The type of the host pointer.
 */
template <typename DevPtr, typename HostPtr>
static inline auto
memcpy_host_to_device(DevPtr* dev_ptr, const HostPtr* host_ptr, size_t bytes)
  -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpy(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice)));
}

/**
 * Copies the given number of bytes of data from the host pointer to the device
 * pointer, asynchronously.
 *
 * \note This will not block on the host until the copy completes.
 *
 * \param  dev_ptr  The device pointer to copy to.
 * \param  host_ptr The host pointer to copy from.
 * \param  bytes    The number of bytes to copy.
 * \tparam DevPtr   The type of the device pointer.
 * \tparam HostPtr  The type of the host pointer.
 */
template <typename DevPtr, typename HostPtr>
static inline auto memcpy_host_to_device_async(
  DevPtr* dev_ptr, const HostPtr* host_ptr, size_t bytes) -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpyAsync(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice)));
}

/**
 * Copies the given number of bytes of data from the host pointer to the device
 * pointer, asynchronously.
 *
 * \note This will not block on the host until the copy completes.
 *
 * \note This requires that the \p host_ptr data be page locked, which should
 *       have been allocated with alloc_pinned(). This overload alows the
 *       stream to be specified, which will allow the copy operation to be
 *       overlapped with operations in other streams, and possibly computation
 *       in the same stream, if suppored.
 *
 * \param  dev_ptr  The device pointer to copy to.
 * \param  host_ptr The host pointer to copy from.
 * \param  bytes    The number of bytes to copy.
 * \param  stream   The stream to perform the copy on.
 * \tparam DevPtr   The type of the device pointer.
 * \tparam HostPtr  The type of the host pointer.
 */
template <typename DevPtr, typename HostPtr>
static inline auto memcpy_host_to_device_async(
  DevPtr* dev_ptr, const HostPtr* host_ptr, size_t bytes, GpuStream stream)
  -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpyAsync(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, stream)));
}

/*==--- [device to host] ---------------------------------------------------==*/

/**
 * Copies the given number of bytes of data from the device pointer to the
 * host pointer.
 *
 * \note This will block on the host until the copy is complete.
 *
 * \param  host_ptr The host pointer to copy from.
 * \param  dev_ptr  The device pointer to copy to.
 * \param  bytes    The number of bytes to copy.
 * \tparam HostPtr  The type of the host pointer.
 * \tparam DevPtr   The type of the device pointer.
 */
template <typename HostPtr, typename DevPtr>
static inline auto
memcpy_device_to_host(HostPtr* host_ptr, const DevPtr* dev_ptr, size_t bytes)
  -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpy(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost)));
}

/**
 * Copies the given number of bytes of data from the device pointer to the
 * host pointer. asynchronously.
 *
 * \note This will not block on the host until the copy is complete.
 *
 * \param  host_ptr The host pointer to copy from.
 * \param  dev_ptr  The device pointer to copy to.
 * \param  bytes    The number of bytes to copy.
 * \tparam HostPtr  The type of the host pointer.
 * \tparam DevPtr   The type of the device pointer.
 */
template <typename HostPtr, typename DevPtr>
static inline auto memcpy_device_to_host_async(
  HostPtr* host_ptr, const DevPtr* dev_ptr, size_t bytes) -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpyAsync(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost)));
}

/**
 * Copies the given number of bytes of data from the device pointer to the
 * host pointer. asynchronously, on the given stream.
 *
 * \note This will not block on the host until the copy is complete.
 *
 * \param  host_ptr The host pointer to copy from.
 * \param  dev_ptr  The device pointer to copy to.
 * \param  bytes    The number of bytes to copy.
 * \param  stream   The stream to perform the copy on.
 * \tparam HostPtr  The type of the host pointer.
 * \tparam DevPtr   The type of the device pointer.
 */
template <typename HostPtr, typename DevPtr>
static inline auto memcpy_device_to_host_async(
  HostPtr* host_ptr, const DevPtr* dev_ptr, size_t bytes, GpuStream stream)
  -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaMemcpyAsync(host_ptr, dev_ptr, bytes, cudaMemcpyDeviceToHost, stream)));
}

/*==--- [allocation device] ------------------------------------------------==*/

/**
 * Allocates the given number of bytes of memory on the device at the location
 * pointed to by the pointer.
 * \param  dev_ptr The device pointer to allocate memory for.
 * \param  bytes   The number of bytes to allocate.
 * \tparam Ptr     The type of the pointer.
 */
template <typename Ptr>
static inline auto allocate_device(Ptr** dev_ptr, size_t bytes) -> void {
  ripple_check_cuda_result(ripple_if_cuda(cudaMalloc((void**)dev_ptr, bytes)));
}

/**
 * Frees the pointer.
 * \param  ptr The pointer to free.
 * \tparam Ptr The type of the pointer to free.
 */
template <typename Ptr>
static inline auto free_device(Ptr* ptr) -> void {
  ripple_check_cuda_result(ripple_if_cuda(cudaFree(ptr)));
}

} // namespace gpu

namespace cpu {

/**
 * Allocates the given number of bytes of page locked (pinned) memory at the
 * location of the host pointer.
 *
 * \todo Add support for pinned allocation if no cuda.
 *
 * \param  host_ptr  The host pointer for the allocated memory.
 * \param  bytes     The number of bytes to allocate.
 * \tparam Ptr       The type of the pointer.
 */
template <typename Ptr>
static inline auto allocate_host_pinned(Ptr** host_ptr, size_t bytes) -> void {
  ripple_check_cuda_result(ripple_if_cuda(
    cudaHostAlloc((void**)host_ptr, bytes, cudaHostAllocPortable)));
}

/**
 * Frees the pointer which was allocated as pinned memory.
 *
 * \param  ptr The pointer to free.
 * \tparam Ptr The type of the pointer to free.
 */
template <typename Ptr>
static inline auto free_host_pinned(Ptr* ptr) -> void {
  ripple_check_cuda_result(ripple_if_cuda(cudaFreeHost(ptr)));
}

} // namespace cpu

} // namespace ripple

#endif // RIPPLE_UTILITY_MEMORY_HPP
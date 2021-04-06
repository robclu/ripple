/**=--- ripple/arch/gpu_utils.hpp -------------------------- -*- C++ -*- ---==**
 *
 *                                Ripple
 *
 *                  Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==------------------------------------------------------------------------==**
 *
 * \file  gpu_utils.hpp
 * \brief This file defines gpu related utilities.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ARCH_GPU_UTILS_HPP
#define RIPPLE_ARCH_GPU_UTILS_HPP

#include <ripple/utility/portability.hpp>

namespace ripple::gpu {

/**
 * Sets the current gpu device to the the device with the given index.
 * \param device_id The id of the device to the set as the current device.
 */
inline auto set_device(uint32_t device_id) noexcept -> void {
  ripple_check_cuda_result(cudaSetDevice(device_id));
}

/**
 * Creates a stream for the device.
 * \param stream The stream to create.
 */
inline auto create_stream(GpuStream* stream) noexcept -> void {
  ripple_check_cuda_result(cudaStreamCreate(stream));
}

/**
 * Ctreates a non-blocking stream for the device.
 */
inline auto create_nonblocking_stream(GpuStream* stream) noexcept -> void {
  ripple_check_cuda_result(
    cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
}

/**
 * Destroys a stream for the device.
 * \param stream The stream to create.
 */
inline auto destroy_stream(GpuStream stream) noexcept -> void {
  ripple_check_cuda_result(cudaStreamDestroy(stream));
}

/**
 * Creates an event on the gpu and starts recording in it.
 * \param stream The stream to create the event in.
 */
inline auto create_event(GpuStream stream) noexcept -> GpuEvent {
  GpuEvent event = nullptr;
  ripple_check_cuda_result(cudaEventCreate(&event));
  ripple_check_cuda_result(cudaEventRecord(event, stream));
  return event;
}

/**
 * Starts recording the event.
 * \param event  The event to record into.
 * \param stream The stream to record on.
 */
inline auto record_event(GpuEvent event, GpuStream stream) noexcept -> void {
  ripple_check_cuda_result(cudaEventRecord(event, stream));
}

/**
 * Creates an event on the gpu and starts recording in it.
 * \param event  A reference to the event to create.
 * \param stream The stream to create the event in.
 */
inline auto destroy_event(GpuEvent event) noexcept -> void {
  ripple_check_cuda_result(cudaEventDestroy(event));
}

/**
 * Synchronizes the given stream.
 *
 * \note This requires that set_device is called prior to this call to ensure
 *       the the correct device is set for the stream.
 *
 * \param  stream The stream to synchronize.
 * \tparam Stream The type of the stream.
 */
inline auto synchronize_stream(GpuStream stream) noexcept -> void {
  ripple_check_cuda_result(cudaStreamSynchronize(stream));
}

/**
 * Synchronizes the given stream by waiting for the event to complete.
 *
 * \note This requires that set_device is called prior to this call to ensure
 *       the the correct device is set for the stream.
 *
 * \param  stream The stream to synchronize.
 * \tparam Stream The type of the stream.
 */
inline auto
synchronize_stream(GpuStream stream, GpuEvent event) noexcept -> void {
  ripple_check_cuda_result(cudaStreamWaitEvent(stream, event));
}

/**
 * Synchronizes the device on the currently active context.
 */
inline auto synchronize_device() noexcept -> void {
  ripple_check_cuda_result(cudaDeviceSynchronize());
}

/**
 * Checks the last gpu error.
 */
inline auto check_last_error() noexcept -> void {
  ripple_check_cuda_result(cudaGetLastError());
}

} // namespace ripple::gpu

#endif // RIPPLE_ARCH_GPU_UTILS_HPP
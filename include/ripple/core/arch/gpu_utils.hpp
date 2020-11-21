//==--- ripple/core/arch/gpu_utils.hpp --------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  gpu_utils.hpp
/// \brief This file defines gpu related utilities.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_GPU_UTILS_HPP
#define RIPPLE_ARCH_GPU_UTILS_HPP

#include "../utility/portability.hpp"

namespace ripple::gpu {

/**
 * Sets the current gpu device to the the device with the given index.
 * \param device_id The id of the device to the set as the current device.
 */
inline auto set_device(uint32_t device_id) noexcept -> void {
  ripple_if_cuda(cudaSetDevice(device_id));
}

/**
 * Creates a stream for the device.
 * \param stream The stream to create.
 */
inline auto create_stream(GpuStream* stream) noexcept -> void {
  ripple_if_cuda(cudaStreamCreate(stream));
}

/**
 * Ctreates a non-blocking stream for the device.
 */
inline auto create_nonblocking_stream(GpuStream* stream) noexcept -> void {
  ripple_if_cuda(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
}

/**
 * Destroys a stream for the device.
 * \param stream The stream to create.
 */
inline auto destroy_stream(GpuStream stream) noexcept -> void {
  ripple_if_cuda(cudaStreamDestroy(stream));
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
  ripple_if_cuda(cudaStreamSynchronize(stream));
}

/**
 * Synchronizes the device on the currently active context.
 */
inline auto synchronize_device() noexcept -> void {
  ripple_if_cuda(cudaDeviceSynchronize());
}

/**
 * Checks the last gpu error.
 */
inline auto check_last_error() noexcept -> void {
  ripple_check_cuda_result(ripple_if_cuda(cudaGetLastError()));
}

} // namespace ripple::gpu

#endif // RIPPLE_ARCH_GPU_UTILS_HPP
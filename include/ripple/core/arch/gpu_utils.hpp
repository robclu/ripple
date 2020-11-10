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

namespace ripple {

/**
 * Sets the current gpu device to the the device with the given index.
 * \param device_id The id of the device to the set as the current device.
 */
inline auto set_device(uint32_t device_id) noexcept -> void {
  ripple_if_cuda(cudaSetDevice(device_id));
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
template <typename Stream>
inline auto synchronize_stream(Stream&& stream) noexcept -> void {
  ripple_if_cuda(cudaStreamSynchronize(stream));
}

} // namespace ripple

#endif // RIPPLE_ARCH_CPU_UTILS_HPP
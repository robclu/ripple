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

#include <ripple/core/utility/portability.hpp>

namespace ripple {

/// Sets the affinity of the thread by binding the context of the current
/// process to thread \p thread_id.
/// Returns false if the operation failed.
/// \param thread_id The index of the thread to bind the context to.

/**
 * Sets the current gpu device to the the device with the given index.
 * \param device_id The id of the device to the set as the current device.
 */
auto set_device(uint32_t device_id) noexcept -> void {
  ripple_if_cuda(cudaSetDevice(device_id));
}

} // namespace ripple

#endif // RIPPLE_ARCH_CPU_UTILS_HPP
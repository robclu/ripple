/**==--- ripple/core/arch/cpu_utils.hpp -------------------- -*- C++ -*- ---==**
 *
 *                                 Ripple
 *
 *                   Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==------------------------------------------------------------------------==//
 *
 * \file  cpu_utils.hpp
 * \brief This file defines cpu related utilities.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ARCH_CPU_UTILS_HPP
#define RIPPLE_ARCH_CPU_UTILS_HPP

#include "../utility/portability.hpp"
#include <cstdint>

/**
 * \todo Add windows support.
 */

#if defined(__linux__)
  #include <sched.h>

  #if defined(__CPU_ISSET)
    #define ripple_cpu_set __CPU_SET
    #define ripple_cpu_zero __CPU_ZERO
    #define ripple_cpu_is_set __CPU_ISSET
  #else
    #define ripple_cpu_set CPU_SET
    #define ripple_cpu_zero CPU_ZERO
    #define ripple_cpu_is_set CPU_ISSET
  #endif // __CPU_ISSET

#endif // __linux__

namespace ripple {

/**
 * Sets the affinity of the thread by binding the context of the current
 * process to thread id.
 * \param thread_id The index of the thread to bind the context to.
 * \return true if the operation succeeded.
 */
inline auto set_affinity(uint32_t thread_id) noexcept -> bool {
#if defined(__linux__)
  cpu_set_t current_thread;
  ripple_cpu_zero(&current_thread);
  ripple_cpu_set(thread_id, &current_thread);
  if (!sched_setaffinity(0, sizeof(current_thread), &current_thread)) {
    return true;
  }
#endif // __linux__

  return false;
}

} // namespace ripple

#endif // RIPPLE_ARCH_CPU_UTILS_HPP

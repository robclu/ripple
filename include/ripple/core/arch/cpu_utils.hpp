//==--- ripple/core/arch/cpu_utils.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cpu_utils.hpp
/// \brief This file defines cpu related utilities.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_CPU_UTILS_HPP
#define RIPPLE_ARCH_CPU_UTILS_HPP

#if defined(__linux__)

#include <sched.h>

#if defined(__CPU_ISSET)
  #define ripple_cpu_set    __CPU_SET
  #define ripple_cpu_zero   __CPU_ZERO
  #define ripple_cpu_is_set __CPU_ISSET
#else
  #define ripple_cpu_set    CPU_SET
  #define ripple_cpu_zero   CPU_ZERO
  #define ripple_cpu_is_set CPU_ISSET
#endif // __CPU_ISSET

#endif // __linux__

namespace ripple {

/// Binds the constext of the current process to thread \p thread_id.
/// Returns false if the operation failed.
/// \param thread_id The index of the thread to bind the context to.
auto bind_context(uint32_t thread_id) -> bool {
  cpu_set_t current_thread;
  ripple_cpu_zero(&current_thread);
  ripple_cpu_set(thread_id, &current_thread);
  if (!sched_setaffinity(0, sizeof(current_thread), &current_thread)) {
    return true;
  }
  return false;
}

} // namespace ripple

#endif // RIPPLE_ARCH_CPU_UTILS_HPP


/**=--- ripple/execution/synchronize.hpp ------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  synchronize.hpp
 * \brief This file implements functionality to synchronize execution.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_EXECUTION_SYNCHRONIZE_HPP
#define RIPPLE_EXECUTION_SYNCHRONIZE_HPP

#include <ripple/utility/portability.hpp>
#include <cooperative_groups.h>

namespace ripple {

/**
 * Synchronizes all active threads in a block on the device.
 *
 * \note There is currently something wrong here causing a compiler error
 *       for sm_70 when the coalesced_threads() are used, so we have to use
 *       the old __syncthreads.
 *
 *       Not sure if this might cause errors since sm_70 can have divergent
 *       branches.
 */
ripple_all inline auto syncthreads() noexcept -> void {
/*
 * For newer architectures, __syncthreads is called on each thread in a block,
 * so if one thread has returned and __syncthreads is called then there will
 * be a deadlock, which is solved by using the coalesced groups.
 */
#if defined(__CUDA__) && __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ != 700
  //__syncthreads();
  // co-operative groups not working on ampere!
  auto g = cooperative_groups::coalesced_threads();
  g.sync();

/*
 * For older architectures, __syncthreads only needs to succeed on one running
 * thread in the block to avoid deadlock, and the coalesced groupd does not
 * work, so here we default to the old syncthreads.
 */
#elif defined(__CUDA__) && defined(__CUDA_ARCH__)
  __syncthreads();
#endif // __CUDACC__
}

/**
 * Does nothing on the host, for now.
 */
// ripple_all inline auto syncthreads() noexcept -> void {}

} // namespace ripple

#endif // RIPPLE_EXECUTION_SYNCHRONIZE_HPP

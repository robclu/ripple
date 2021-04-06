/**=--- ripple/utiliy/spinlock.hpp ------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  spinlock.hpp
 * \brief This file defines a struct to store gpu information.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_UTILITY_SPINLOCK_HPP
#define RIPPLE_UTILITY_SPINLOCK_HPP

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>

namespace ripple {

/**
 * A small spinlock implementation.
 */
struct Spinlock {
 private:
  // clang-format off
  /** Defines the state of a free lock. */
  static constexpr uint8_t free   = 0;
  /** Defines the state of a locked lock. */
  static constexpr uint8_t locked = 1;
  // clang-format

  /**
   * A struct which can be used to spin for a certain number of iterations and
   * then sleep if the iteration count has been reached.
   */
  struct Sleeper {
    /** Defines the max number of spins before sleeping. */
    static constexpr uint32_t max_spins = 4000;

    /**
     * Puts the thread to sleep for a duration which is __usually__ less than
     * the minimum sleep time for the kernel, so the kernel will __usually__
     * schedule this thread to sleep for the minimum duration.
     */
    static auto sleep() noexcept -> void {
      using namespace std::chrono_literals;
      // Sleep for an amount which will let the kernel schedule us for the min
      // duration, which is usually somewhere between 1 and 10ms.
      std::this_thread::sleep_for(200us);
    }

    /**
     * Causes the CPU to wait if the spin count is less than the maximum number
     * of spins, otherwise sleeps for the kernel min sleep duration.
     */
    auto wait() noexcept -> void {
      if (spincount_ < max_spins) {
        spincount_++;
        // Essentially _mm_pause() and a memory barrier in one instruction.
        // Just to make sure that there is no memory reordering which might
        // be the case if the compiler decided to move things around.
        // The pause prevents speculative loads from causing pipeline clears
        // due to memory ordering mis-speculation.
        asm volatile("pause" ::: "memory");
        return;
      }
      sleep();
    }

   private:
    uint32_t spincount_ = 0; //!< Number of spins.
  };

 public:
  /**
   * Tries to lock the spinlock, returning true if the lock succeded.
   * \return true of the lock is successful.
   */
  auto try_lock() noexcept -> bool {
    return __sync_bool_compare_and_swap(&lock_, free, locked);
  }

  /**
   * Locks the spinlock. This will block until the lock is acquired.
   */
  auto lock() noexcept -> void {
    Sleeper sleeper;
    while (!__sync_bool_compare_and_swap(&lock_, free, locked)) {
      do {
        sleeper.wait(); // Wait until CAS might succeed.
      } while (lock_);
    }
  }

  /**
   * Unlocks the spinlock.
   */
  auto unlock() noexcept -> void {
    // Memory barrier so that we can write the lock to the unlocked state.
    asm volatile("" ::: "memory");
    lock_ = free;
  }

 private:
  /* TODO: Compare performance vs using std::atomic<uint8_t> */
  uint8_t lock_ = free; //!< Lock state for the spinlock.
};

} // namespace ripple

#endif
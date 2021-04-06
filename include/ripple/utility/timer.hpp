/**=--- ripple/utiliy/timer.hpp ---------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  timer.hpp
 * \brief This file defines a simple timer class.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_UTILITY_TIMER_HPP
#define RIPPLE_UTILITY_TIMER_HPP

#include <chrono>

namespace ripple {

/**
 * The Timer struct is a simple class which starts on construction, and stops
 * with a call to elapsed. The timer is also resettable.
 */
struct Timer {
  // clang-format off
  /** Defines the type of the clock. */
  using Clock     = std::chrono::high_resolution_clock;
  /** Defines the type of the time point. */
  using TimePoint = typename Clock::time_point;
  /** Defines the type of the duration. */
  using Duration  = std::chrono::duration<double>;
  // clang-format on

  /** Convesion factor from micro to milliseconds. */
  static constexpr double microsecs_to_millisecs = 1000.0;

  /**
   * Constructor which starts the timer.
   */
  Timer() noexcept {
    reset();
  }

  /**
   * Resets the timer by restarting the clock.
   */
  auto reset() noexcept -> void {
    start_ = Clock::now();
  }

  /**
   * Gets the elapsed time since the start of the clock, in micro seconds.
   * \param reset_timer If the timer must be reset.
   * \return The number of microseconds since the timer was started.
   */
  auto elapsed(bool reset_timer = false) noexcept -> double {
    TimePoint curr      = Clock::now();
    Duration  time_span = std::chrono::duration_cast<Duration>(curr - start_);

    if (reset_timer) {
      reset();
    }
    return time_span.count();
  }

  /**
   * Gets the number of elapsed milliseconds since the start of the timer, and
   * optionally resets the timer.
   * \param reset_timer If the timer must be reset.
   * \return The number of milliseconds since the timer started.
   */
  auto elapsed_msec(bool reset_timer = false) noexcept -> double {
    return elapsed(reset_timer) * microsecs_to_millisecs;
  }

 private:
  TimePoint start_; //!< Time point at the start of the timer.
};

} // namespace ripple

#endif // RIPPLE_UTILITY_TIMER_HPP

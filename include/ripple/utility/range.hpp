/**=--- ripple/utility/rangle.hpp -------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  range.hpp
 * \brief This file defines the implementation of an iterator which allows
 *        python like ranges.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_UTILITY_RANGE_HPP
#define RIPPLE_UTILITY_RANGE_HPP

#include "portability.hpp"
#include "type_traits.hpp"

namespace ripple {

/**
 * The Range class defines a utility class which allows a simpler syntax for
 *  looping --  a python like range based for loop. It is wrapped range
 *  by the ``range()`` function, and the inteded usage is:
 *
 *  ~~~cpp
 *  // i = [0 ... 100)
 *  for (auto i : range(100)) {
 *    // Use i
 *  }
 *
 *  // i = [34 .. 70)
 *  for (auto i : range(34, 70)) {
 *    // Use i
 *  }
 *
 *  // i = [23 .. 69) step = 3
 *  for (auto i : range(23, 69, 3)) {
 *    // Use i
 *  }
 *  ~~~
 *
 *  The type of the range elements are defined by the type passed to the range
 *  creation function (i.e pass a float to get a range of floats):
 *
 *  ~~~cpp
 *  for (auto i : range(0.1f, 0.6f, 0.1f)) {
 *    // Use i
 *  }
 *  ~~~
 *
 *  \tparam T The type of the range data.
 */
template <typename T>
class Range {
  /** Defines the type of the iterator data. */
  using Value = std::decay_t<T>;

  /**
   * The Iterator class defines an iterator for iterating over a range.
   * \tparam IsConst If the iterator is constant.
   */
  template <bool IsConst>
  struct IteratorImpl {
    // clang-format off
    /** Defines the type of the iterator. */
    using Self      = IteratorImpl<IsConst>;
    /** Defines the type of the iterator data. */
    using IterValue = std::decay_t<T>;
    /** Defines the type of a reference. */
    using Reference = std::conditional_t<IsConst, const T&, T&>;
    /** Defines the type of a pointer. */
    using Pointer   = std::conditional_t<IsConst, const T*, T*>;
    // clang-format on

    /**
     * Sets the value of the iterator and the step size.
     * \param value   The value for the iterator.
     * \param step    The size of the steps for the iterator.
     */
    ripple_all constexpr IteratorImpl(
      IterValue value, IterValue step) noexcept
    : value_(value), step_(step) {}

    /**
     * Overload of increment operator to move the iterator forward by the step
     * size. This overload is for the postfix operator and returns the old
     * value of the iterator.
     */
    ripple_all constexpr auto operator++() noexcept -> Self {
      Self i = *this;
      value_ += step_;
      return i;
    }

    /**
     * Overload of increment operator to move the iterator forward by the step
     * size. This overload is for the prefix operator and returns the updated
     * iterator.
     */
    ripple_all constexpr auto operator++(int) noexcept -> Self {
      value_ += step_;
      return *this;
    }

    /** Returns a reference to the value of the iterator. */
    ripple_all constexpr auto operator*() noexcept -> Reference {
      return value_;
    }

    /** Returns a pointer to the value of the iterator. */
    ripple_all constexpr auto operator->() noexcept -> Pointer {
      return &value_;
    }

    /**
     * Overload of the equality operator to check if two iterators are
     * equivalent.
     *
     * \return true if the value of this iterator is greater than the or equal
     *         to the value of the other iterator, false otherwise.
     *
     * \param other The other iterator to compare with.
     */
    ripple_all constexpr auto
    operator==(const Self& other) noexcept -> bool {
      return value_ >= other.value_;
    }

    /**
     * Overload of the inequality operator to check if two iterators are
     * not equivalent.
     *
     * \return true if the value of this iterator is less than the value of
     *         other, false otherwise.
     *
     * \param other The other iterator to compare with.
     */
    ripple_all constexpr auto
    operator!=(const Self& other) noexcept -> bool {
      return value_ < other.value_;
    }

   private:
    IterValue value_; //!< The current value of the range iterator.
    IterValue step_;  //!< The step size of the range iterator.
  };

  Value min_;  //!< The minimum value in the range.
  Value max_;  //!< The maximum value in the range.
  Value step_; //!< The step size for iterating over the range.

 public:
  // clang-format off
  /** Defines the type of a constant iterator. */
  using ConstIterator = IteratorImpl<true>;
  /** Defines the type of a non-constant iterator. */
  using Iterator      = IteratorImpl<false>;
  // clang-format on

  /** Creates the range.
   * \param min  The minimum (start) value for the range.
   * \param max  The maximum (end) value for the range.
   * \param step The step size for the range.
   */
  ripple_all constexpr Range(Value min, Value max, Value step) noexcept
  : min_(min), max_(max), step_(step) {}

  /** Gets a non constant iterator to the beginning of the range. */
  ripple_all constexpr auto begin() noexcept -> Iterator {
    return Iterator{min_, step_};
  }

  /** Gets a non constant iterator to the end of the range. */
  ripple_all constexpr auto end() noexcept -> Iterator {
    return Iterator{max_, step_};
  }

  /** Gets a constant iterator to the beginning of the range. */
  ripple_all constexpr auto begin() const -> ConstIterator {
    return ConstIterator{min_, step_};
  }

  /** Gets a non constant iterator to the end of the range. */
  ripple_all constexpr auto end() const -> ConstIterator {
    return ConstIterator{max_, step_};
  }
};

/*==--- [functions] --------------------------------------------------------==*/

/**
 * Creates a range from 0 to end, using a step size of 1.
 * \param  end The end value for the range
 * \tparam T   The type of the range data.
 */
template <typename T>
ripple_all constexpr inline auto range(T end) noexcept -> Range<T> {
  return Range<T>(T{0}, static_cast<T>(end), T{1});
}

/**
 * Creates a range from start to end, using a step size of step.
 * \param  start The staring value of the range.
 * \param  end   The end value for the range
 * \param  step  The step size of the range.
 * \tparam T     The type of the range data.
 */
template <typename T>
ripple_all constexpr inline auto
range(T start, T end, T step = T{1}) noexcept -> Range<T> {
  return Range<T>(start, end, step);
}

} // namespace ripple

#endif // RIPPPLE_UTILITY_RANGE_HPP

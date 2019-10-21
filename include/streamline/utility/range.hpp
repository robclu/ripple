//==--- streamline/utility/range.hpp ----------------------- -*- C++ -*- ---==//
//
//                                Streamline
//                                
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  range.hpp
/// \brief This file defines the implementation of an iterator which allows
///        python like ranges.
//
//==------------------------------------------------------------------------==//

#ifndef STREAMLINE_ITERATOR_RANGE_HPP
#define STREAMLINE_ITERATOR_RANGE_HPP

#include "portability.hpp"
#include "type_traits.hpp"

namespace streamline {

/// The Range class defines a utility class which allows a simpler syntax for
/// looping --  a python like range based for loop. It is wrapped range
/// by the ``range()`` function, and the inteded usage is:
/// 
/// ~~~cpp
/// // i = [0 ... 100)
/// for (auto i : range(100)) {
///   // Use i
/// }
/// 
/// // i = [34 .. 70)
/// for (auto i : range(34, 70)) {
///   // Use i
/// }
/// 
/// // i = [23 .. 69) step = 3
/// for (auto i : range(23, 69, 3)) {
///   // Use i 
/// }
/// 
/// The type of the range elements are defined by the type passed to the range
/// creation function (i.e pass a float to get a range of floats):
/// 
/// ~~~cpp
/// for (auto i : range(0.1f, 0.6f, 0.1f)) {
///   // Use i
/// }
/// 
/// \tparam T The type of the range data.
template <typename T>
class Range {
  /// Defines the type of the iterator data.
  using value_t = std::decay_t<T>;

  /// The Iterator class defines an iterator for iterating over a range.
  /// \tparam IsConst If the iterator is constant.
  template <bool IsConst>
  struct Iterator {
    /// Defines the type of the iterator.
    using self_t      = Iterator;
    /// Defines the type of the iterator data.
    using value_t     = std::decay_t<T>;
    /// Defines the type of a reference.
    using reference_t = std::conditional_t<IsConst, const T&, T&>;
    /// Defines the type of a pointer.
    using pointer_t   = std::conditional_t<IsConst, const T*, T*>;

    //==--- [construction] -------------------------------------------------==//

    /// Sets the value of the iterator and the step size.
    /// \param[in]  value   The value for the iterator.
    /// \param[in]  step    The size of the steps for the iterator.
    streamline_host_device constexpr Iterator(value_t value, value_t step)
    : _value(value), _step(step) {}

    //==--- [operator overloads] -------------------------------------------==//

    /// Overload of increment operator to move the iterator forward by the step
    /// size. This overload is for the postfix operator and returns the old
    /// value of the iterator.
    streamline_host_device constexpr self_t operator++() { 
      self_t i = *this; _value += _step; return i;
    }

    /// Overload of increment operator to move the iterator forward by the step
    /// size. This overload is for the prefix operator and returns the updated
    /// iterator.
    streamline_host_device constexpr self_t operator++(int) {
      _value += _step; return *this;
    }

    /// Returns a reference to the value of the iterator.
    streamline_host_device constexpr reference_t operator*() {
      return _value;
    }

    /// Returns a pointer to the value of the iterator.
    streamline_host_device constexpr pointer_t operator->() {
      return &_value;
    }

    /// Overload of the equality operator to check if two iterators are
    /// equivalent. This returns true if the value of this iterator is greater
    /// than the or equal to the value of \p other.
    /// \param[in]  other   The other iterator to compare with.
    streamline_host_device constexpr bool operator==(const self_t& other) {
      return _value >= other._value;
    }

    /// Overload of the inequality operator to check if two iterators are
    /// not equivalent. This returns true if the value of this iterator is less
    /// than the value of \p other.
    /// \param[in]  other   The other iterator to compare with.
    streamline_host_device constexpr bool operator!=(const self_t& other) {
      return _value < other._value;
    }

   private:
    value_t _value; //!< The current value of the range iterator.
    value_t _step;  //!< The step size of the range iterator.
  };

  value_t _min;  //!< The minimum value in the range.
  value_t _max;  //!< The maximum value in the range.
  value_t _step; //!< The step size for iterating over the range.

 public:
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of a constant iterator.
  using const_iterator_t = Iterator<true>;
  /// Defines the type of a non-constant iterator.
  using iterator_t       = Iterator<false>;

  //==--- [construction] ---------------------------------------------------==//
  
  /// Creates the range.
  /// \param[in]  min   The minimum (start) value for the range.
  /// \param[in]  max   The maximum (end) value for the range.
  /// \param[in]  step  The step size for the range.
  streamline_host_device constexpr Range(value_t min, value_t max, value_t step)
  : _min(min), _max(max), _step(step) {}

  /// Gets a non constant iterator to the beginning of the range.
  streamline_host_device auto begin() {
    return iterator_t{_min, _step};
  }
  
  /// Gets a non constant iterator to the end of the range.
  streamline_host_device auto end() {
   return iterator_t{_max, _step};   
  }

  /// Gets a constant iterator to the beginning of the range.
  streamline_host_device auto begin() const {
    return const_iterator_t(_min, _step);
  }
  
  /// Gets a non constant iterator to the end of the range.
  streamline_host_device auto end() const {
   return const_iterator_t(_max, _step);   
  }
};

//==--- [functions] --------------------------------------------------------==//

/// Creates a range from 0 to \p end, using a step size of 1.
/// \param[in]  end   The end value for the range
/// \tparam     T     The type of the range data.
template <typename T>
streamline_host_device constexpr inline auto range(T end) -> Range<T> {
  return Range<T>(T{0}, static_cast<T>(end), T{1});
}

/// Creates a range from \p start to \p end, using a step size of \p step.
/// \param[in]  start The staring value of the range.
/// \param[in]  end   The end value for the range
/// \param[in]  step  The step size of the range.
/// \tparam     T     The type of the range data.
template <typename T>
streamline_host_device constexpr inline auto range(
  T start, T end, T step = T{1}
) -> Range<T> {
  return Range<T>(start, end, step);
}

} // namespace streamline

#endif // FLUIDITY_ITERATOR_RANGE_HPP

//==--- ripple/core/functional/kernel/invoke_utils_.cuh ---- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_utils_.cuh
/// \brief This file implements functionality to invoke a pipeline on various
///        container objects on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP

#include <ripple/core/boundary/load_boundary.hpp>
#include <ripple/core/execution/synchronize.hpp>
#include <ripple/core/execution/detail/thread_index_impl_.hpp>
#include <ripple/core/execution/thread_index.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>

namespace ripple::kernel::cuda::util {

//==--- [iterator shifting] ------------------------------------------------==//

/// Dummy class which can be consumed when applying a function to a variadic
/// pack.
struct Consumer {};

/// Shifts the \p it iterator in the \p dim dimension by \p amount. This
/// overload is only enabled when the type of the \p it __is__ an iterator.
/// \param  it       The iterator to shift.
/// \param  amount   The amount to shift the iterator by.
/// \param  dim      The dimension to shift in.
/// \tparam Iterator The type of the iterator.
/// \tparam Dim      The type of the dimension specifier.
template <typename Iterator, typename Dim, iterator_enable_t<Iterator> = 0>
ripple_device_only auto shift_iter_in_dim(
  Iterator& it, size_t amount, Dim&& dim
)-> Consumer {
  it.shift(dim, amount);
  return Consumer();
}

/// Shifts the \p it iterator in the \p dim dimension by \p amount. This
/// overload is only enabled when the type of the \p it is __not__ an iterator.
/// \param  it       The iterator to shift.
/// \param  amount   The amount to shift the iterator by.
/// \param  dim      The dimension to shift in.
/// \tparam Iterator The type of the iterator.
/// \tparam Dim      The type of the dimension specifier.
template <typename Dim, typename Iterator, non_iterator_enable_t<Iterator> = 0>
ripple_device_only auto shift_iter_in_dim(
  Iterator& it, size_t amount, Dim&& dim 
) -> Consumer {
  return Consumer();
}

/// This function simply does nothing, and results in a no op. It's purpose is
/// to apply an operation to a variadic pack, consuming the result of each
/// operation.
/// \param  args The arguments to consume.
/// \tparam Args The type of the arguments.
template <typename... Args>
ripple_device_only auto consume(Args&&... args) -> void {}

/// Shifts the iterator by the global index if the global index is in the range
/// of the iterator, and shifts any of the \p args if they are iterators.
///
/// It returns true if the iterator is in range.
///
/// \param  it       The iterator to shift.
/// \tparam Iterator The type of the iterator.
template <typename Iterator, typename... Args>
ripple_device_only auto shift_in_range_global(
  Iterator& it, Args&&... args 
) -> bool {
  bool in_range = true;
  constexpr auto dims = iterator_traits_t<Iterator>::dimensions;
  unrolled_for<dims>([&] (auto dim) {
    const auto idx = global_idx(dim);
    if (in_range && idx < it.size(dim)) {
      it.shift(dim, idx);
      if constexpr (sizeof...(Args) > 0) {
        consume(shift_iter_in_dim(args, idx, dim)...);
      }
    } else {
      in_range = false;
    }
  });
  return in_range;
}

/// Shifts the iterator by the global index if the global index is in the range
/// of the iterator, and the \p shared_it by the thread index and it's padding.
/// Additionally, it shifts any of the \p args by the global index if any of the
/// \p args are iterator types.
///
/// It returns true if the iterator is in range.
///
/// \param  it             The iterator to shift.
/// \param  shared_it      The shared memory iterator to shift.
/// \tparam Iterator       The type of the iterator.
/// \tparam SharedIterator The type of the shared memory iterator.
template <typename Iterator, typename SharedIterator, typename... Args>
ripple_device_only auto shift_in_range(
  Iterator& it, SharedIterator& sit, Args&&... args
) -> bool {
  bool in_range       = true;
  constexpr auto dims = iterator_traits_t<Iterator>::dimensions;
  unrolled_for<dims>([&] (auto dim) {
    const auto idx = global_idx(dim);
    if (in_range && idx < it.size(dim)) {
      it.shift(dim, idx);
      sit.shift(dim, thread_idx(dim) + sit.padding());
      if constexpr (sizeof...(Args) > 0) {
        consume(shift_iter_in_dim(args, idx, dim)...);
      }
    } else {
      in_range = false;
    }
  });
  return in_range;
}

//==--- [iterator data setting] --------------------------------------------==//

/// Sets the data for \p it_b to that of \p it_b is the types being iterated
/// over are either the same, or convertible to each other.
/// \param  it_a      The iterator whose data to use to set.
/// \param  it_b      The iterator whose data to set.
/// \param  IteratorA The type of the first iterator.
/// \tparam IteratorB The type of the second iterator.
template <typename IteratorA, typename IteratorB>
ripple_device_only auto set_iter_data(const IteratorA& it_a, IteratorB& it_b) 
-> void {
  using it_a_t = std::decay_t<decltype(*it_a)>;
  using it_b_t = std::decay_t<decltype(*it_b)>;

  constexpr auto must_set = 
    std::is_same_v<it_a_t, it_b_t> || std::is_convertible_v<it_a_t, it_b_t>;

  if constexpr (must_set) {
    *it_b = *it_a;
  }
}

/// Sets the boundary (padding) data for \p it_b using \p it_b.
/// \param  it_a      The iterator whose data to use to set.
/// \param  it_b      The iterator whose data to set.
/// \param  IteratorA The type of the first iterator.
/// \tparam IteratorB The type of the second iterator.
template <typename IteratorA, typename IteratorB>
ripple_device_only auto set_iter_boundary(IteratorA& it_a, IteratorB& it_b)
-> void {
  if (it_b.padding() > 0) {
    constexpr auto dims = iterator_traits_t<IteratorB>::dimensions;
    load_internal_boundary<dims>(it_a, it_b);
  }
}

} // namespace ripple::kernel::cuda::util

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP

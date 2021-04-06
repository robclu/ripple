/**=--- ripple/kernel/detail/invoke_utils_.cuh ------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  invoke_utils_.cuh
 * \brief This file implements functionality to invoke a pipeline on various
 *        container objects on the device.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__CUH
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__CUH

#include <ripple/container/block.hpp>
#include <ripple/container/block_traits.hpp>
#include <ripple/execution/synchronize.hpp>
#include <ripple/execution/detail/thread_index_impl_.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/iterator/iterator_traits.hpp>
#include <ripple/padding/load_padding.hpp>

namespace ripple::kernel::gpu::util {

/*==--- [iterator data setting] --------------------------------------------==*/

/**
 * Sets the data for it_b to that of it_a if the types being iterated over
 * are either the same, or convertible to each other.
 *  \param  it_a      The iterator whose data to use to set.
 *  \param  it_b      The iterator whose data to set.
 *  \param  IteratorA The type of the first iterator.
 *  \tparam IteratorB The type of the second iterator.
 */
template <typename IteratorA, typename IteratorB>
ripple_device auto
set_iter_data(const IteratorA& it_a, IteratorB& it_b) -> void {
  using UnderlyingA = std::decay_t<decltype(*it_a)>;
  using UnderlyingB = std::decay_t<decltype(*it_b)>;

  constexpr auto must_set = std::is_same_v<UnderlyingA, UnderlyingB> ||
                            std::is_convertible_v<UnderlyingA, UnderlyingB>;

  if constexpr (must_set) {
    *it_b = *it_a;
  }
}

/**
 * Sets the padding data for it_b using  it_a.
 *  \param  it_a      The iterator whose data to use to set.
 *  \param  it_b      The iterator whose data to set.
 *  \param  IteratorA The type of the first iterator.
 *  \tparam IteratorB The type of the second iterator.
 */
template <typename IteratorA, typename IteratorB>
ripple_device auto
set_iter_boundary(IteratorA& it_a, IteratorB& it_b) noexcept -> void {
  constexpr auto dims = iterator_traits_t<IteratorB>::dimensions;
  //  load_internal_boundary<dims>(it_a, it_b);

  /* Used to call load_internal_boundary(), however, ther reguster usage was
   * very high, so now we are just shifting instead. It's simple enough that
   * it produces faster code. */
  const int32_t pad = it_b.padding();
  if constexpr (dims == 2) {
    it_a.shift(dimx(), -pad);
    it_b.shift(dimx(), -pad);
    it_a.shift(dimy(), -pad);
    it_b.shift(dimy(), -pad);
    *it_b = *it_a;
    it_a.shift(dimx(), 2 * pad);
    it_b.shift(dimx(), 2 * pad);
    *it_b = *it_a;
    it_a.shift(dimy(), 2 * pad);
    it_b.shift(dimy(), 2 * pad);
    *it_b = *it_a;
    it_a.shift(dimx(), -2 * pad);
    it_b.shift(dimx(), -2 * pad);
    *it_b = *it_a;
    it_a.shift(dimx(), pad);
    it_b.shift(dimx(), pad);
    it_a.shift(dimy(), -pad);
    it_b.shift(dimy(), -pad);
  }
}

} // namespace ripple::kernel::gpu::util

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__CUH
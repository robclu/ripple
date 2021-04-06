/**=--- ripple/algorithm/reduce.hpp ------------------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  reduce.hpp
 * \brief This file implements a reduction on a block.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ALGORITHM_REDUCE_HPP
#define RIPPLE_ALGORITHM_REDUCE_HPP

#include "kernel/reduce_cpp_.hpp"
#include "kernel/reduce_cuda_.cuh"
#include <ripple/container/host_block.hpp>

namespace ripple {

/**
 * \note There is supposedly hardware support for certain predicates, see:
 *
 *  #collectives-cg-reduce
 *
 * In the CUDA programming guide, however, it's been tested on sm_80 and sm_86
 * and there is very little difference in performance compared to the
 * implementations below, which work with both the host and device reduction
 * versions, so these are preferred. Perhaps there is more difference in
 * performance for different use cases than those which were tested.
 *
 */

/*==--- [predicates] -------------------------------------------------------==*/

/**
 * Functor which can be used with the reduction to perform a reduction sum.
 */
struct SumReducer {
  /**
   * Adds the data pointed to by the one iterator to the other.
   * \param  a An iterator to the data to add to.
   * \param  b An iterator to the data to add.
   * \tparam T    The type of the iterator.
   */
  template <typename T>
  ripple_host_device auto inplace(T& a, const T& b) const noexcept -> void {
    *a += *b;
  }

  /**
   * Adds the data \p a and \p b and returns the result.
   * \param  a The first data to add.
   * \param  b The second data to add.
   * \tparam T The type of the data.
   * \return The addition of the two elements.
   */
  template <typename T>
  ripple_host_device auto
  operator()(const T& a, const T& b) const noexcept -> T {
    return T{a + b};
  }
};

/**
 * Functor which can be used with the reduction to perform a reduction
 * subtraction.
 */
struct SubtractionReducer {
  /**
   * Subtracts the data pointed to by the one iterator from the other.
   * \param  a An iterator to the data to subtract from.
   * \param  b An iterator to the data to suctract with.
   * \tparam T The type of the iterator.
   */
  template <typename T>
  ripple_host_device auto inplace(T& a, const T& b) const noexcept -> void {
    *a -= *b;
  }

  /**
   * Subtracts the data \p b from \p a and returns the result.
   * \param  a The first data to subtract from.
   * \param  b The second data to subtract with.
   * \tparam T The type of the data.
   * \return The subtraction of b from a.
   */
  template <typename T>
  ripple_host_device auto
  operator()(const T& a, const T& b) const noexcept -> T {
    return a - b;
  }
};

/**
 * Functor which can be used with the reduction to find a max value over a
 * dataset.
 */
struct MaxReducer {
  /**
   * Sets the \p a data to the max of \p a and \p b.
   * \param  a The component to set to the max.
   * \param  b The compoennt to comapare with.
   * \tparam T The type of the iterator.
   */

  template <typename T>
  ripple_host_device auto inplace(T& a, const T& b) const noexcept -> void {
    *a = std::max(*a, *b);
  }

  /**
   * Returns the max of \p a and \p b
   * \param  a The first input for comparison.
   * \param  b The second input for comparison
   * \tparam T The type of the data.
   * \return The max of a and b.
   */
  template <typename T>
  ripple_host_device auto
  operator()(const T& a, const T& b) const noexcept -> T {
    return std::max(a, b);
  }
};

/**
 * Functor which can be used with the reduction to find a min value over a
 * dataset.
 */
struct MinReducer {
  /**
   * Sets the \p a data to the min of \p a and \p b.
   * \param  a The component to set to the min.
   * \param  b The compoennt to comapare with.
   * \tparam T The type of the iterator.
   */
  template <typename T>
  ripple_host_device auto operator()(T& a, const T& b) const noexcept -> void {
    *a = std::min(*a, *b);
  }

  /**
   * Returns the min of \p a and \p b
   * \param  a The first input for comparison.
   * \param  b The second input for comparison
   * \tparam T The type of the data.
   * \return The min of a and b.
   */
  template <typename T>
  ripple_host_device auto
  operator()(const T& a, const T& b) const noexcept -> T {
    return std::min(a, b);
  }
};

/*==--- [interface] --------------------------------------------------------==*/

/**
 * Reduces the \p block using the \p pred, returning the result.
 *
 * The pred must have the following overloads form:
 *
 * ~~~.cpp
 * ripple_host_device auto inplace(T<&> into, <const> T<&> from) const ->
 *   -> void {}
 *
 * and
 *
 * ripple_host_device auto operator()(const T<&> a, <const> T<&> b) const
 *  -> T {}'
 * ~~~
 *
 * where T is an iterator over the type T, or a pointer, for thee inplace
 * version, and is a reference of value for the operator overload version.
 *
 * For the inplace version, the predicate *must modify* the data pointed to by
 * the `into` iterator (first argument), as appropriate, using the `from`
 * iterator. Modifying the `from` iterator may  cause unexpectd results, so
 * make it const if it should not be modified.
 *
 * \note Even though the signatures are different, the CUDA coop groups reduce
 *       implementation doesn't choose the correct one if we make both
 *       operator() overloads.
 *
 * This overload is for device blocks.
 *
 * \param  block The block to reduce.
 * \param  pred  The predicate for the reduction.
 * \tparam T     The type of the data in the block.
 * \tparam Dims  The number of dimensions in the block.
 * \tparam Pred  The type of the predicate.
 */
template <typename T, size_t Dims, typename Pred>
auto reduce(const DeviceBlock<T, Dims>& block, Pred&& pred) noexcept -> T {
  return kernel::gpu::reduce(block, ripple_forward(pred));
}

/**
 * Reduces the \p block using the \p pred, returning the result.
 *
 * The pred must have the following overloads form:
 *
 * ~~~.cpp
 * ripple_host_device auto inplace(T<&> into, <const> T<&> from) const ->
 *   -> void {}
 *
 * and
 *
 * ripple_host_device auto operator()(const T<&> a, <const> T<&> b) const
 *  -> T {}'
 * ~~~
 *
 * where T is an iterator over the type T, or a pointer, for thee inplace
 * version, and is a reference of value for the operator overload version.
 *
 * For the inplace version, the predicate *must modify* the data pointed to by
 * the `into` iterator (first argument), as appropriate, using the `from`
 * iterator. Modifying the `from` iterator may  cause unexpectd results, so
 * make it const if it should not be modified.
 *
 * \note Even though the signatures are different, the CUDA coop groups reduce
 *       implementation doesn't choose the correct one if we make both
 *       operator() overloads.
 *
 * This overload is for host blocks.
 *
 * \param  block The block to reduce.
 * \param  pred  The predicate for the reduction.
 * \tparam T     The type of the data in the block.
 * \tparam Dims  The number of dimensions in the block.
 * \tparam Pred  The type of the predicate.
 */
template <typename T, size_t Dims, typename Pred>
auto reduce(const HostBlock<T, Dims>& block, Pred&& pred) noexcept -> T {
  return kernel::cpu::reduce(block, ripple_forward(pred));
}

} // namespace ripple

#endif // RIPPLE_ALGORITHM_REDUCE_HPP

//==--- ripple/core/algorithm/kernel/reduce_cpp_.hpp ------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce_cpp_.hpp
/// \brief This file implements functionality to reduce a multi dimensional
///        block on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALGORITHM_KERNEL_REDUCE_CPP__HPP
#define RIPPLE_ALGORITHM_KERNEL_REDUCE_CPP__HPP

#include "../../iterator/iterator_traits.hpp"

namespace ripple::kernel::cpu {
namespace detail {

/**
 * Forward declaration of a struct to perform a reduction in the given number
 * of dimensions.
 * \tparam Dims The number of dimensions for the reduction.
 */
template <size_t Dims>
class Reducer;

/**
 * Specialization of the reducer struct for the zero (x) dimension.
 */
template <>
class Reducer<0> {
  /** Defines the dimension to reduce over. */
  static constexpr auto dim = ripple::dimx();

 public:
  /**
   * Performs a reduction in the x dimension. This does not reduce the first
   * element, and assumes that the result initially has the value of the first
   * element.
   *
   * \param  it       The iterator to invoke the callable on.
   * \param  result   The iterator which points to the result.
   * \param  pred     The predicate to apply to each of the elements.
   * \param  args     Additional arguments for the predicate.
   * \tparam Dim      The dimension to reduce over.
   * \tparam Iterator The type of the iterators.
   * \tparam Pred     The type of the predicate.
   * \tparam Args     The type of the predicate additional arguments.
   */
  template <
    typename Iterator1,
    typename Iterator2,
    typename Pred,
    typename... Args>
  ripple_host_device static auto reduce(
    Iterator1&& it, Iterator2&& result, Pred&& pred, Args&&... args) noexcept
    -> void {
    static_assert(is_iterator_v<Iterator1>, "Reduction requires an iterator!");
    static_assert(is_iterator_v<Iterator2>, "Reduction requires an iterator!");

    constexpr size_t sub_iters = 4;

    // Not reducing the first element, hence -1.
    const auto elements  = it.size(dim) - 1;
    const auto iters     = elements / sub_iters;
    const auto rem_iters = elements % sub_iters;

    auto other = it;
    for (size_t i = 0; i < iters; ++i) {
      unrolled_for<sub_iters>([&](auto j) {
        const size_t idx = i * sub_iters + j;
        other            = it.offset(dim, idx + 1);
        pred.inplace(result, other);
      });
    }

    for (size_t i = 0; i < rem_iters; ++i) {
      const size_t idx = iters * sub_iters + i;
      other            = it.offset(dim, idx + 1);
      pred.inplace(result, other);
    }
  }
};

/**
 * Struct to define a reducer for a domain with the given number of dimensions.
 * \tparam Dim The number of dimensions in the domain.
 */
template <size_t Dim>
class Reducer {
  /** Defines the type of the reducer for the next dimension. */
  using NextReducer = Reducer<Dim - 1>;

  /** Defines the value of the dimension to reduce in. */
  static constexpr auto dim = Dim == 1 ? ripple::dimy() : ripple::dimz();

 public:
  /**
   * Performs a reduction in the given dimension by performing a reduction on
   * each of the (rows/planes) in the Dim - 1 dimension.
   *
   * \param  it        The iterator to reduce the data over.
   * \param  result    The iterator to reduce into.
   * \param  pred      The predicate to apply to each of the elements.
   * \param  args      Additional arguments for the predicate.
   * \tparam Iterator1 The type of the first iterator.
   * \tparam Iterator2 The type of the second iterator.
   * \tparam Pred      The type of the predicate.
   * \tparam Args      The type of the predicate additional arguments.
   */
  template <
    typename Iterator1,
    typename Iterator2,
    typename Pred,
    typename... Args>
  ripple_host_device static auto reduce(
    Iterator1&& it, Iterator2&& result, Pred&& pred, Args&&... args) noexcept
    -> void {
    static_assert(is_iterator_v<Iterator1>, "Reduction requires an iterator!");
    static_assert(is_iterator_v<Iterator2>, "Reduction requires an iterator!");

    // Reduce the first row/plane:
    NextReducer::reduce(
      it, result, ripple_forward(pred), ripple_forward(args)...);
    for (size_t i = 0; i < it.size(dim) - 1; ++i) {
      auto next = it.offset(dim, i + 1);
      // Next dimension doesn't do the first element:
      pred.inplace(result, next);

      NextReducer::reduce(
        next, result, ripple_forward(pred), ripple_forward(args)...);
    }
  }
};

} // namespace detail

/**
 * Reduces the block using the predicate.
 *
 * \param  block The block to invoke the callable on.
 * \param  pred  The predicate for the reduction.
 * \param  args  Arguments for the predicate.
 * \tparam T     The type of the data in the block.
 * \tparam Dims  The number of dimensions in the block.
 * \tparam Pred  The type of the predicate.
 * \tparam Args  The type of the arguments for the invocation.
 * \return The result of the reduction.
 */
template <typename T, size_t Dims, typename Pred, typename... Args>
auto reduce(
  const HostBlock<T, Dims>& block, Pred&& pred, Args&&... args) noexcept {
  using Reducer = detail::Reducer<Dims - 1>;

  // Store the inital value, and then reduce into the first iterated value.
  auto it         = block.begin();
  auto init_value = *it;
  Reducer::reduce(it, it, ripple_forward(pred), ripple_forward(args)...);

  // Get results and reset the first element, since it has been reduced into.
  auto result = *it;
  *it         = init_value;
  return result;
}

} // namespace ripple::kernel::cpu

#endif // RIPPLE_ALGORITHM_KERNEL_REDUCE_CPP__HPP
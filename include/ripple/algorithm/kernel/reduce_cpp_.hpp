//==--- ripple/algorithm/kernel/reduce_cuda_.cuh ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce_cuda_.cuh
/// \brief This file implements functionality to reduce a multi dimensional
///        block on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALGORITHM_KERNEL_INVOKE_CPP__HPP
#define RIPPLE_ALGORITHM_KERNEL_INVOKE_CPP__HPP

#include <ripple/functional/invoke.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/iterator/iterator_traits.hpp>

namespace ripple::kernel {
namespace detail {

/// Forward declaration of a struct to perform a reduction in Dims dimensions.
/// \tparam Dims The number of dimensions for the reduction.
template <std::size_t Dims> class Reducer;

/// Specialization of the reducer struct for the zero (x) dimension.
template <> class Reducer<0> {
  /// Defines the dimension to reduce over.
  static constexpr auto dim = ripple::dim_x;
 public:
  /// Performs a reduction in the x dimension. This does not reduce the first
  /// element, and assumes that \p result initially has the value of the first
  /// element.
  ///
  /// \param  it       The iterator to invoke the callable on.
  /// \param  result   The iterator which points to the result.
  /// \param  pred     The predicate to apply to each of the elements.
  /// \param  args     Additional arguments for the predicate.
  /// \tparam Dim      The dimension to reduce over.
  /// \tparam Iterator The type of the iterators.
  /// \tparam Pred     The type of the predicate.
  /// \tparam Args     The type of the predicate additional arguments.
  template <
    typename    Iterator1, 
    typename    Iterator2, 
    typename    Pred     , 
    typename... Args
  >
  ripple_host_device static auto reduce(
    Iterator1&& it, Iterator2&& result, Pred&& pred, Args&&... args
  ) -> void {
    constexpr auto sub_iters = 4;

    // Not reducing the first element, hence -1.
    const auto elements      = it.size(dim) - 1;
    const auto iters         = elements / sub_iters;
    const auto rem_iters     = elements % sub_iters;

    auto other = it;
    for (auto i : range(iters)) {
      unrolled_for<sub_iters>([&] (auto j) {
        const auto idx  = i * sub_iters + j;
        other = it.offset(dim, idx + 1);
        pred(result, other);
      });
    }

    for (auto i : range(rem_iters)) {
      const auto idx = iters * sub_iters + i;
      other = it.offset(dim, idx + 1);
      pred(result, other);
    }
  }
};

/// Struct to define a reducer for a domain with Dim dimensions.
/// \tparam Dim The number of dimensions in the domain.
template <std::size_t Dim> 
class Reducer {
  //==--- [aliases] --------------------------------------------------------==//
  /// Defines the type of the reducer for the next dimension.
  using next_reducer_t = Reducer<Dim - 1>;
  //==--- [constants] ------------------------------------------------------==//
  /// Defines the value of the dimension to reduce in.
  static constexpr auto dim = Dim == 1 ? ripple::dim_y : ripple::dim_z;
 public:
  /// Performs a reduction in the Dim dimension by performing a reduction on
  /// each of the (rows/planes) in the Dim - 1 dimension.
  ///
  /// \param  it        The iterator to reduce the data over.
  /// \param  result    The iterator to reduce into.
  /// \param  pred      The predicate to apply to each of the elements.
  /// \param  args      Additional arguments for the predicate.
  /// \tparam Iterator1 The type of the first iterator.
  /// \tparam Iterator2 The type of the second iterator.
  /// \tparam Pred      The type of the predicate.
  /// \tparam Args      The type of the predicate additional arguments.
  template <
    typename    Iterator1,
    typename    Iterator2, 
    typename    Pred     ,
    typename... Args
  >
  ripple_host_device static auto reduce(
    Iterator1&& it, Iterator2&& result, Pred&& pred, Args&&... args
  ) -> void {
    // Reduce the first row/plane:
    next_reducer_t::reduce(
      it, result, std::forward<Pred>(pred), std::forward<Args>(args)...
    );
    for (auto i : range(it.size(dim) - 1)) {
      auto next = it.offset(dim, i + 1);
      // Next dimension doesn't do the first element:
      pred(result, next);

      next_reducer_t::reduce(
        next, result, std::forward<Pred>(pred), std::forward<Args>(args)...
      );
    }
  }
};

} // namespace detail

/// Reduces the \p block using the \p pred.
///
/// \param  block The block to invoke the callable on.
/// \param  pred  The predicate for the reduction.
/// \param  args  Arguments for the predicate.
/// \tparam T     The type of the data in the block.
/// \tparam Dims  The number of dimensions in the block.
/// \tparam Pred  The type of the predicate.
/// \tparam Args  The type of the arguments for the invocation.
template <typename T, std::size_t Dims, typename Pred, typename... Args>
auto reduce(const HostBlock<T, Dims>& block, Pred&& pred, Args&&... args) {
  using reducer_t = detail::Reducer<Dims - 1>;

  // Store the inital value, and then reduce into the first iterated value.
  auto it         = block.begin();;
  auto init_value = *it;
  reducer_t::reduce(
    it, it, std::forward<Pred>(pred), std::forward<Args>(args)...
  );

  // Get results and reset the first element, since it has been reduced into.
  auto result = *it;
  *it = init_value;;
  return result;
}

} // namespace ripple::kernel

#endif // RIPPLE_ALGORITHM_KERNEL_REDUCE_CPP__HPP


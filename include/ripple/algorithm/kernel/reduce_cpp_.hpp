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
template <std::size_t Dims> struct Reducer;

/// Specialization of the reducer struct for the zero (x) dimension.
template <> struct Reducer<0> {
  private:
  /// Defines the dimension to reduce over.
  static constexpr auto dim = ripple::dim_x;

 public:
  /// Performs a reduction in the x dimension.
  ///
  /// \param  it       The iterator to invoke the callable on.
  /// \param  result   The result to reduce into.
  /// \param  pred     The predicate to apply to each of the elements.
  /// \param  args     Additional arguments for the predicate.
  /// \tparam Dim      The dimension to reduce over.
  /// \tparam Iterator The type of the iterator.
  /// \tparam T        The type of the data in the block.
  /// \tparam Pred     The type of the predicate.
  /// \tparam Args     The type of the predicate additional arguments.
  template <typename Iterator, typename T, typename Pred, typename... Args>
  ripple_host_device static auto reduce(
    Iterator&&  it    ,
    T&          result,
    Pred&&      pred  ,
    Args&&...   args  
  ) -> void {
    constexpr auto sub_iters = 4;
    const auto iters         = it.size(dim) / sub_iters;
    const auto rem_iters     = it.size(dim) % sub_iters;

    for (auto i : range(iters)) {
      unrolled_for<sub_iters>([&] (auto j) {
        const auto idx = i * sub_iters + j;
        pred(result, *it.offset(dim, idx));
      });
    }

    for (auto i : range(rem_iters)) {
      const auto idx = iters * sub_iters + i;
      pred(result, *it.offset(dim, idx));
    }
  }
};


/// Struct to define a reducer for a domain with Dim dimensions.
/// \tparam Dim The number of dimensions in the domain.
template <std::size_t Dim> 
struct Reducer {
 private:
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
  /// \param  it       The iterator to invoke the callable on.
  /// \param  result   The result to reduce into.
  /// \param  pred     The predicate to apply to each of the elements.
  /// \param  args     Additional arguments for the predicate.
  /// \tparam Iterator The type of the iterator.
  /// \tparam T        The type of the data in the block.
  /// \tparam Pred     The type of the predicate.
  /// \tparam Args     The type of the predicate additional arguments.
  template <typename Iterator, typename T, typename Pred, typename... Args>
  ripple_host_device static auto reduce(
    Iterator&& it    ,
    T&         result,
    Pred&&     pred  ,
    Args&&...  args  
  ) -> void {
    for (auto i : range(it.size(dim))) {
      next_reducer_t::reduce(
        it.offset(dim, i)       ,
        result                  ,
        std::forward<Pred>(pred),
        std::forward<Args>(args)...    
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
  auto it         = block.begin();
  auto result     = *it;
  reducer_t::reduce(
    it                      ,
    result                  ,
    std::forward<Pred>(pred),
    std::forward<Args>(args)...
  );
  return result - *it;
}

} // namespace ripple::kernel

#endif // RIPPLE_ALGORITHM_KERNEL_REDUCE_CPP__HPP


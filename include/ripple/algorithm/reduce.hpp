//==--- ripple/algorithm/reduce.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce.hpp
/// \brief This file implements a reduction on a block.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALGORITHM_REDUCE_HPP
#define RIPPLE_ALGORITHM_REDUCE_HPP

#include "kernel/reduce_cpp_.hpp"
#include "kernel/reduce_cuda_.cuh"
#include <ripple/container/host_block.hpp>

namespace ripple {

/// Reduces the \p block using the \p pred, returning the result. 
///
/// The pred must have the following form:
///
/// ~~~.cpp
/// ripple_host_device auto (T<&> into, <const> T<&> from) const -> void {
///   //...
/// };
/// ~~~
///
/// where T is an iterator, and therefore the reference is optional. The
/// predicate must modify the data pointed to by the `into` iterator, as
/// appropriate, using the `from` iterator. Modifying the `from` iterator may 
/// cause unexpectd results.
///
/// This overload is for device blocks.
///
/// \param  block The block to reduce.
/// \param  pred  The predicate for the reduction.
/// \param  args  Arguments for the predicate.
/// \tparam T     The type of the data in the block.
/// \tparam Dims  The number of dimensions in the block.
/// \tparam Pred  The type of the predicate.
/// \tparam Args  The type of the arguments for the invocation.
template <typename T, std::size_t Dims, typename Pred, typename... Args>
auto reduce(const DeviceBlock<T, Dims>& block, Pred&& pred, Args&&... args) {
  return kernel::cuda::reduce(
    block, std::forward<Pred>(pred), std::forward<Args>(args)...
  );
}

/// Reduces the \p block using the \p pred, returning the result. 
///
/// The pred must have the form:
///
/// ~~~.cpp
/// ripple_host_device auto (T<&> into, <const> T<&> from) const -> void {
///   //...
/// };
/// ~~~
///
/// where T is an iterator, and therefore the reference is optional. The
/// predicate must modify the data pointed to by the `into` iterator, as
/// appropriate, using the `from` iterator. Modifying the `from` iterator may 
/// cause unexpectd results.
///
/// This overload is for host blocks.
///
/// \param  block The block to reduce.
/// \param  pred  The predicate for the reduction.
/// \param  args  Arguments for the predicate.
/// \tparam T     The type of the data in the block.
/// \tparam Dims  The number of dimensions in the block.
/// \tparam Pred  The type of the predicate.
/// \tparam Args  The type of the arguments for the invocation.
template <typename T, std::size_t Dims, typename Pred, typename... Args>
auto reduce(const HostBlock<T, Dims>& block, Pred&& pred, Args&&... args) {
  return kernel::reduce(
    block, std::forward<Pred>(pred), std::forward<Args>(args)...
  );
}

} // namespace ripple

#endif // RIPPLE_ALGORITHM_REDUCE_HPP


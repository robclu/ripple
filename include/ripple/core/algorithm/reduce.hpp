//==--- ripple/core/algorithm/reduce.hpp ------------------------ -*- C++ -*- ---==//
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
#include <ripple/core/container/host_block.hpp>

namespace ripple {

//==--- [predicates] -------------------------------------------------------==//

/// Functor which can be used with the reduction to perform a reduction sum.
struct SumReducer {
  /// Adds the data pointed to by the \p from iterator to the data pointed to by
  /// \p into iterator.
  /// \param  into An iterator to the data to add to.
  /// \param  from An iterator to the data to add.
  /// \tparam T    The type of the iterator.
  template <typename T>
  ripple_host_device auto operator()(T& into, const T& from) const -> void {
    *into += *from;
  }
};

/// Functor which can be used with the reduction to perform a reduction
/// subtraction.
struct SubtractionReducer {
  /// Subtracts the data pointed to by the \p with iterator from the data
  /// pointed to by \p from iterator.
  /// \param  from An iterator to the data to subtract from.
  /// \param  with An iterator to the data to suctract with.
  /// \tparam T    The type of the iterator.
  template <typename T>
  ripple_host_device auto operator()(T& from, const T& with) const -> void {
    *from -= *with;
  }
};

/// Functor which can be used with the reduction to find a max value over a
/// dataset.
struct MaxReducer {
  /// Sets the \p a data to the max of \p a and \p b.
  /// \param  a The component to set to the max.
  /// \param  b The compoennt to comapare with.
  /// \tparam T The type of the iterator.
  template <typename T>
  ripple_host_device auto operator()(T& a, const T& b) const -> void {
    *a = std::max(*a, *b);
  }
};

/// Functor which can be used with the reduction to find a min value over a
/// dataset.
struct MinReducer {
  /// Sets the \p a data to the min of \p a and \p b.
  /// \param  a The component to set to the min.
  /// \param  b The compoennt to comapare with.
  /// \tparam T The type of the iterator.
  template <typename T>
  ripple_host_device auto operator()(T& a, const T& b) const -> void {
    *a = std::min(*a, *b);
  }
};

//==--- [interface] --------------------------------------------------------==//

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
/// where T is an iterator, over the type T, which is the type of the data in
/// the block. Because an iterator is passing, the reference is optional. The
/// predicate must modify the data pointed to by the `into` iterator (first
/// argument), as appropriate, using the `from` iterator. Modifying the 
/// `from` iterator may  cause unexpectd results, so make it const if it should
/// not be modified.
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
auto reduce(const DeviceBlock<T, Dims>& block, Pred&& pred, Args&&... args) 
-> T {
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
auto reduce(const HostBlock<T, Dims>& block, Pred&& pred, Args&&... args) 
-> T {
  return kernel::reduce(
    block, std::forward<Pred>(pred), std::forward<Args>(args)...
  );
}

} // namespace ripple

#endif // RIPPLE_ALGORITHM_REDUCE_HPP


//==--- ripple/execution/execution_size.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  execution_size.hpp
/// \brief This file contains functionality for computing the size of an
///        execution space for a kernel.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_EXECUTION_SIZE_HPP
#define RIPPLE_EXECUTION_EXECUTION_SIZE_HPP

#include <ripple/container/device_block.hpp>
#include <ripple/utility/type_traits.hpp>
#include <cmath>
#include <tuple>

namespace ripple::exec {

/// Computes the number of blocks given that there are \p elements in the
/// dimension and \p threads in the dimension for the block.
/// \param  elements    The number of elements in the dimension.
/// \param  max_threads The number of threads in the dimension.
/// \tparam T           The type of the elements.
/// \tparam U           The type of the max threads.
template <typename T, typename U>
auto get_dim_num_blocks(T elements, U threads) -> std::size_t {
  return std::max(static_cast<std::size_t>(
      std::ceil(static_cast<double>(elements) / static_cast<double>(threads))
    ),
    std::size_t{1}
  );
}

/// Computes the number of threads given that there are \p elements in the
/// dimension, and \p max_threads maximum number of threads for the dimension.
/// If \p elements < \p max_threads, then this returns \p max_threads, otherwise
/// it returns \p elements.
/// \param  elements    The number of elements in the dimension.
/// \param  max_threads The maximum number of threads for the dimension.
/// \tparam T           The type of the elements.
/// \tparam U           The type of the max threads.
template <typename T, typename U>
auto get_dim_num_threads(T elements, U max_threads) -> std::size_t {
  return std::min(
    static_cast<std::size_t>(elements), static_cast<std::size_t>(max_threads)
  );
}

/// Returns the number of threads and blocks required to launch a kernel for the
/// \p block on the device. This overload is only enabled if the \p block is one
/// dimensional.
/// \param  block The block to generate the execution size for.
/// \tparam T     The data type for the block.
/// \tparam Dims  The number of dimensions for the block.
template <typename T, std::size_t Dims, dim_1d_enable_t<Dims> = 0>
auto get_exec_size(const DeviceBlock<T, Dims>& block) 
-> std::tuple<dim3, dim3> { 
  constexpr auto threads_x = 1024;
  const auto     elems_x   = block.size(dim_x);

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);
  
  threads.x = get_dim_num_threads(elems_x, threads_x);
  blocks.x  = get_dim_num_blocks(elems_x, threads.x);

  return std::make_tuple(threads, blocks);
}

/// Returns the number of threads and blocks required to launch a kernel for the
/// \p block on the device. This overload is only enabled if the \p block is two
/// dimensional.
/// \param  block The block to generate the execution size for.
/// \tparam T     The data type for the block.
/// \tparam Dims  The number of dimensions for the block.
template <typename T, std::size_t Dims, dim_2d_enable_t<Dims> = 0>
auto get_exec_size(const DeviceBlock<T, Dims>& block) 
-> std::tuple<dim3, dim3> { 
  constexpr auto threads_x = 32;
  constexpr auto threads_y = 32;
  const auto     elems_x   = block.size(dim_x);
  const auto     elems_y   = block.size(dim_y);

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, threads_x);
  threads.y = get_dim_num_threads(elems_y, threads_y);
  blocks.x  = get_dim_num_blocks(elems_x, threads.x);
  blocks.y  = get_dim_num_blocks(elems_y, threads.y);

  return std::make_tuple(threads, blocks);
}

/// Returns the number of threads and blocks required to launch a kernel for the
/// \p block on the device. This overload is only enabled if the \p block is
/// three dimensional.
/// \param  block The block to generate the execution size for.
/// \tparam T     The data type for the block.
/// \tparam Dims  The number of dimensions for the block.
template <typename T, std::size_t Dims, dim_3d_enable_t<Dims> = 0>
auto get_exec_size(const DeviceBlock<T, Dims>& block) 
-> std::tuple<dim3, dim3> {
  constexpr auto threads_x = 8;
  constexpr auto threads_y = 8;
  constexpr auto threads_z = 8;
  const auto     elems_x   = block.size(dim_x);
  const auto     elems_y   = block.size(dim_y);
  const auto     elems_z   = block.size(dim_z);

  auto threads = dim3(1, 1, 1);
  auto blocks  = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(elems_x, threads_x);
  threads.y = get_dim_num_threads(elems_y, threads_y);
  threads.z = get_dim_num_threads(elems_z, threads_z);
  blocks.x  = get_dim_num_blocks(elems_x, threads.x);
  blocks.y  = get_dim_num_blocks(elems_y, threads.y);
  blocks.z  = get_dim_num_blocks(elems_z, threads.z);

  return std::make_tuple(threads, blocks);
}

} // namespace ripple::exec

#endif // RIPPLE_EXECUTION_EXECUTION_SIZE_HPP

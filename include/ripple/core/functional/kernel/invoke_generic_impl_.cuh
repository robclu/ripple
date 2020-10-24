//==--- ../kernel/invoke_generic_impl_.cuh ----------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_generic_impl_.cuh
/// \brief This file provides the implementation of a generic invoke function
///        to invoke a callable type with a variadic number of arguments.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENERIC_IMPL__CUH
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_GENERIC_IMPL__CUH

#include "invoke_utils_.cuh"
#include <ripple/core/algorithm/max_element.hpp>
#include <ripple/core/execution/dynamic_execution_params.hpp>
#include <ripple/core/execution/execution_params.hpp>
#include <ripple/core/execution/execution_size.hpp>
#include <ripple/core/execution/execution_traits.hpp>
#include <ripple/core/execution/synchronize.hpp>
#include <ripple/core/execution/detail/thread_index_impl_.hpp>
#include <ripple/core/execution/thread_index.hpp>

namespace ripple::kernel::gpu {

/**
 * Stream wrapper class which holds a cuda stream and a parameter which defines
 * if it is valid.
 */
struct Stream {
  bool         set = false; //!< If the stream has been set.
  cudaStream_t stream;      //!< The stream being wrapped.
};

/**
 * Holds information for shared memory.
 * \tparam Dims       The number of dimensions for the shared memory.
 * \tparam SharedData The type of the data for the shared memory.
 * \tparam Space      The space which defines the shared memory.
 */
template <size_t Dims, typename SharedData, typename Space>
struct SharedMemInfo {
  // clang-format off
  /** Defines the type of the allocator for the shared data type. */
  using Alloc = typename layout_traits_t<SharedData>::Allocator;
  /** Defines the type of the iterator over the shared data. */
  using Iter  = BlockIterator<SharedData, Space>;
  // clang-format on

  /**
   * Sets the space which defines the shared memory region.
   * \param  exec_params The execution params which define hold the space.
   * \tparam ParamsImpl  The type of the execution params implementation.
   */
  template <typename ParamsImpl>
  ripple_host_device auto
  set_space(const ExecParams<ParamsImpl>& exec_params) noexcept -> void {
    unrolled_for<Dims>([&](auto dim) { space[dim] = exec_params.size(dim); });
    space.padding() = padding;
  }

  /**
   * Make an iterator over the shared memory region which starts at the data
   * pointer.
   * \param  data A pointer to the data for the space.
   * \tparam T    The type of the data pointer.
   * \return An iterator over the space.
   */
  template <typename T>
  ripple_host_device auto make_iterator(T* data) const noexcept -> Iter {
    return Iter{Alloc::create(
                  static_cast<void*>(static_cast<char*>(data) + offset), space),
                space};
  }

  size_t offset          = 0; //!< Offset into shared memory.
  size_t mem_requirement = 0; //!< Amount of mem required for the space.
  size_t padding         = 0; //!< Padding required for the iterator.
  Space  space;               //!< The space for the shared memory.
};

/**
 * Wrapper struct to represent void shared memory information.
 */
struct VoidInfo {};

/** True if the template parameter is VoidInfo. */
template <typename T>
static constexpr bool is_void_info_v =
  std::is_same_v<VoidInfo, std::decay_t<T>>;

/*==--- [enables] ----------------------------------------------------------==*/

/**
 * Defines a valid type if both types are iterators.
 * \tparam A The first type to check if is an iterator.
 * \tparam B The second type to check if is an iterator.
 */
template <typename A, typename B>
using both_iters_enable_t =
  std::enable_if_t<is_iterator_v<A> && is_iterator_v<B>, int>;

/**
 * Defines a valid type if the condition that both are iterators is false.
 * \tparam A The first type to check if is an iterator.
 * \tparam B The second type to check if is an iterator.
 */
template <typename A, typename B>
using not_both_iters_enable_t =
  std::enable_if_t<!(is_iterator_v<A> && is_iterator_v<B>), int>;

/**
 * Defines a valid type if only the first type is an iterator.
 * \tparam A The first type to check if is an iterator.
 * \tparam B The second type to check if is not an iterator.
 */
template <typename A, typename B>
using only_first_iter_enable_t =
  std::enable_if_t<is_iterator_v<A> && !is_iterator_v<B>, int>;

/**
 * Defines a valid type if only the second type is an iterator.
 * \tparam A The first type to check if is not an iterator.
 * \tparam B The second type to check if is an iterator.
 */
template <typename A, typename B>
using only_second_iter_enable_t =
  std::enable_if_t<!is_iterator_v<A> && is_iterator_v<B>, int>;

/**
 * Defines a valid type if both types are not iterators.
 * \tparam A The first type to check if is not an iterator.
 * \tparam B The second type to check if is not an iterator.
 */
template <typename A, typename B>
using neither_iters_enable_t =
  std::enable_if_t<!is_iterator_v<A> && !is_iterator_v<B>, int>;

/**
 * Defines a valid type if the template parameter is a VoidInfo type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using void_info_enable_t = std::enable_if_t<is_void_info_v<T>, int>;

/**
 * Defines a valid type if the template parameter is not a VoidInfo type.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_void_info_enable_t = std::enable_if_t<!is_void_info_v<T>, int>;

/*==--- [block size] -------------------------------------------------------==*/

/**
 * Gets the size of dimension from the block.
 *
 * \note This overload is only enabled which the type is block enabled.
 *
 * \param  block The block to get the size of.
 * \param  dim   The dimension to get the size of.
 * \tparam T     The type of the block.
 * \tparam Dim   The type of the dimension specifier.
 */
template <typename T, typename Dim, block_enabled_t<T> = 0>
auto get_block_size(T&& block, Dim&& dim) noexcept -> size_t {
  return block_enabled_traits_t<T>::dimensions > dim
           ? block.size(std::forward<Dim>(dim))
           : size_t{0};
}

/**
 * Gets the size of dimension from the block.
 *
 * \note This overload is only enabled which the type is not block enabled,
 *       so just returns zero.
 *
 * \param  block The block to get the size of.
 * \param  dim   The dimension to get the size of.
 * \tparam T     The type of the block.
 * \tparam Dim   The type of the dimension specifier.
 */
template <typename T, typename Dim, non_block_enabled_t<T> = 0>
auto get_block_size(T&& block, Dim&& dim) noexcept -> size_t {
  return 0;
}

/**
 * Gets the size of dimension from the shared wrapper.
 *
 * \note This overload is for a shared wrapper, and forwards the wrapped type
 *       to the other implementations to get the size.
 *
 * \param  wrapper The block to get the size of.
 * \param  dim     The dimension to get the size of.
 * \tparam T       The type of the wrapped type.
 * \tparam Dim     The type of the dimension specifier.
 */
template <typename T, typename Dim>
auto get_block_size(SharedWrapper<T>& wrapper, Dim&& dim) noexcept -> size_t {
  return get_block_size(wrapper.wrapped, static_cast<Dim&&>(dim));
}

/*==--- [shared memory ops] ------------------------------------------------==*/

/**
 * Gets a shared memory itertor if the template parameter is not a VoidInfo
 * type.
 *
 * \note This overload is for VoidInfo types, and simply returns a reference to
 *       the input time.
 *
 * \param  t    The type to get a reference to.
 * \param  data A pointer to the shared memory buffer.
 * \tparam T    The type to get a shared interator over.
 */
template <typename T, void_info_enable_t<T> = 0>
ripple_device_only auto make_shared_iterator(T& t, void* data) noexcept -> T& {
  return t;
}

/**
 * Gets a shared memory itertor if the template parameter is not a VoidInfo
 * type.
 *
 * \note This overload is for non VoidInfo types, and makes an iterator over
 *       the shared memory buffer.
 *
 * \param  shared_info The shared info to create the iterator from.
 * \param  data        A pointer to the shared memory buffer.
 * \tparam T           The type to get a shared interator over.
 */
template <typename T, non_void_info_enable_t<T> = 0>
ripple_device_only auto
make_shared_iterator(T& shared_info, void* data) noexcept -> typename T::Iter {
  return shared_info.make_iterator(data);
}

/**
 * Creates shared memory infromation for the case that the type is not a block
 * type, and just creates void shared memory information. This becomes a no-op.
 *
 * \param  offset  A reference to the offset into the shared memory.
 * \param  stream  A reference to the stream.
 * \param  params The execution parameters.
 * \param  data   The wrapped data to put into shared memory.
 * \tparam Dims   The number of dimensions for the execution.
 * \tparam T      The type of the data for shared memory.
 * \tparam Params The implementation type of the execution params.
 * \return Void information for the shared memory space.
 */
template <size_t Dims, typename T, typename Params, non_block_enabled_t<T> = 0>
auto create_shared_info(
  size_t&                   offset,
  Stream&                   stream,
  const ExecParams<Params>& params,
  T&                        data) noexcept -> VoidInfo {
  return VoidInfo{};
}

/**
 * Creates shared memory infromation for the case that the type is a block
 * type but doesn't require shared memory, so just creates void shared memory
 * information. This will set the stream if it hasn't been set.
 *
 * \param  offset     A reference to the offset into the shared memory.
 * \param  stream     A reference to the stream to possibly set.
 * \param  params     The execution parameters.
 * \param  data       The wrapped data to put into shared memory.
 * \tparam Dims       The number of dimensions for the execution.
 * \tparam T          The type of the data for shared memory.
 * \tparam ParamsImpl The implementation type of the execution params.
 * \return Void information for the shared memory space.
 */
template <size_t Dims, typename T, typename Params, block_enabled_t<T> = 0>
auto create_shared_info(
  size_t& offset, Stream& stream, const ExecParams<Params>& params, T& data)
  -> VoidInfo {
  if (!stream.set) {
    stream.stream = data.stream();
    stream.set    = true;
    cudaSetDevice(data.gpu_id);
  }
  return VoidInfo{};
}

/**
 * Creates shared memory infromation for the case that the type is not a block
 * type but is wrapped for shared memory, so creates the shared memory
 * information.
 *
 * \param  offset      A reference to the offset into the shared memory.
 * \param  stream      A reference to the stream to possibly set.
 * \param  exec_params The execution parameters.
 * \param  data        The wrapped data to put into shared memory.
 * \tparam Dims        The number of dimensions for the execution.
 * \tparam T           The type of the data for shared memory.
 * \tparam ParamsImpl  The implementation type of the execution params.
 * \return Shared memory information for the type.
 */

template <size_t Dims, typename T, typename Params, non_block_enabled_t<T> = 0>
auto create_shared_info(
  size_t&                   offset,
  const Stream&             stream,
  const ExecParams<Params>& params,
  SharedWrapper<T>&         data) noexcept {
  using Traits = layout_traits_t<T>;
  using Alloc  = typename Traits::Allocator;

  /**
   * Here we need to:
   *  - Determine the amount of memory which will allow allocation of an
   *    iterator over type T.
   *  - Compute the offset into the shared space.
   *  - Determine the amount of padding for the shared space.
   */
  constexpr size_t align      = alignof(T);
  const size_t     shared_pad = padding(data);
  const size_t     padding    = (align - (offset % align)) % align;
  const size_t     start      = offset + padding;
  const size_t     elements   = params.template size<Dims>(shared_pad);
  const size_t     size       = Alloc::allocation_size(elements);
  offset                      = start + size;

  using SharedInfoType = SharedMemInfo<Dims, T, DynamicMultidimSpace<Dims>>;
  SharedInfoType info{start, size, shared_pad};
  info.set_space(params);

  return info;
}

/**
 * Creates shared memory infromation for the case that the type is a block type
 * and is wrapped for shared memory.
 *
 * \param  offset      A reference to the offset into the shared memory.
 * \param  stream      A reference to the stream to possibly set.
 * \param  exec_params The execution parameters.
 * \param  data        The wrapped data to put into shared memory.
 * \tparam Dims        The number of dimensions for the execution.
 * \tparam T           The type of the data for shared memory.
 * \tparam ParamsImpl  The implementation type of the execution params.
 * \return Shared memory information for the type.
 */
template <size_t Dims, typename T, typename Params, block_enabled_t<T> = 0>
auto create_shared_info(
  size_t&                   offset,
  Stream&                   stream,
  const ExecParams<Params>& params,
  SharedWrapper<T>&         data) noexcept {
  using BlockEnabledTraits = block_enabled_traits_t<T>;
  using SharedType         = typename BlockEnabledTraits::Value;
  using Traits             = layout_traits_t<SharedType>;
  using Alloc              = typename Traits::Allocator;

  if (!stream.set) {
    stream.stream = data.wrapped.stream();
    stream.set    = true;
    cudaSetDevice(data.wrapped.gpu_id);
  }

  /**
   * Here we need to:
   *  - Determine the amount of memory which will allow allocation of an
   *    iterator over type T.
   *  - Compute the offset into the shared space.
   *  - Determine the amount of padding for the shared space.
   */
  constexpr size_t align      = alignof(SharedType);
  const size_t     shared_pad = padding(data);
  const size_t     padding    = (align - (offset % align)) % align;
  const size_t     start      = offset + padding;
  const size_t     elements   = params.template size<Dims>(shared_pad);
  const size_t     size       = Alloc::allocation_size(elements);
  offset                      = start + size;

  /**
   * Shared memory type is the value type, we keep the same layout even if it's
   * possible to change it, since this has shown to be more performant!
   */
  using SharedInfoType =
    SharedMemInfo<Dims, SharedType, DynamicMultidimSpace<Dims>>;
  SharedInfoType info{start, size, shared_pad};
  info.set_space(params);
  return info;
}

/*===---- [exec sizes] -----------------------------------------------------==*/

/**
 * Gets the number of threads and thread blocks given the sizes of the execution
 * parameter sizes and the global sizes of the space.
 *
 * \param  exec_params  Parameters for the execution space.
 * \param  dim_sizes    The dimension sizes of the domain.
 * \tparam ExecParams   The type of the execution parameters.
 * \return A tuple with the thread and blocks sizes required to perform
 *         computation on the domain with the requested execution paramteres.
 */
template <typename ExecParams>
auto get_execution_sizes(
  const ExecParams& params, const std::array<size_t, 3>& dim_sizes) noexcept
  -> std::tuple<dim3, dim3> {
  auto threads = dim3(1, 1, 1), blocks = dim3(1, 1, 1);

  threads.x = get_dim_num_threads(dim_sizes[dim_x], params.size(dim_x));
  threads.y = get_dim_num_threads(dim_sizes[dim_y], params.size(dim_y));
  threads.z = get_dim_num_threads(dim_sizes[dim_z], params.size(dim_z));
  blocks.x  = get_dim_num_blocks(dim_sizes[dim_x], threads.x);
  blocks.y  = get_dim_num_blocks(dim_sizes[dim_y], threads.y);
  blocks.z  = get_dim_num_blocks(dim_sizes[dim_z], threads.z);

  return std::make_tuple(threads, blocks);
}

/*==--- [iterator offsetting] ----------------------------------------------==*/

/**
 * Offsets the iterators if both the shared and global types are iterators.
 * If they are valid, then valid_count is increased, otherwise it is not.
 * Also, if they are valid, the shared data is set from the global data, and
 * if the shared iterator has padding, the padding is also loaded from the
 * global iterator.
 *
 * \note This overload is only enabled when both the shared and global
 *       paramters are iterators.
 *
 * \param  shared      An iterator over shared memory.
 * \param  global      An iterator over global memory.
 * \param  iter_count  The number of iterators which have been offset.
 * \param  valid_count The number of iterators which are valid.
 * \tparam Shared      The type of the shared memory iterator.
 * \tparam Global      The type of the global memory iterator.
 * \return A tuple with the modified iterators.
 */
template <
  typename Shared,
  typename Global,
  both_iters_enable_t<Shared, Global> = 0>
ripple_host_device auto offset_iters(
  Shared&& shared,
  Global&  global,
  uint8_t& iter_count,
  uint8_t& valid_count) noexcept -> Tuple<Shared, Global&> {
  iter_count++;
  bool valid = true;
  unrolled_for<iterator_traits_t<Shared>::dimensions>([&](auto dim) {
    if (thread_idx(dim) >= shared.size(dim) || !global.is_valid(dim)) {
      valid = false;
    } else {
      shared.shift(dim, thread_idx(dim) + shared.padding());
      global.shift(dim, global_idx(dim));
    }
  });
  if (valid) {
    *shared = *global;
    if (shared.padding() != 0) {
      gpu::util::set_iter_boundary(global, shared);
    }
    valid_count++;
  }
  return Tuple<Shared, Global&>{shared, global};
}

/**
 * Offsets the iterator and set its value to be that of the other parameter. If
 * the shared iteraor is valid, then valid_count is increased, otherwise it is
 * not.
 *
 * \note This overload is only enabled when the shared type is an iterator.
 *
 * \param  shared      An iterator over shared memory.
 * \param  other       The data to set the shared iterator with.
 * \param  iter_count  The number of iterators which have been offset.
 * \param  valid_count The number of iterators which are valid.
 * \tparam Shared      The type of the shared memory iterator.
 * \tparam Other        The type of the other data.
 * \return A tuple with the modified iterator.
 */
template <
  typename Shared,
  typename Other,
  only_first_iter_enable_t<Shared, Other> = 0>
ripple_host_device auto offset_iters(
  Shared&& shared,
  Other&&  other,
  uint8_t& iter_count,
  uint8_t& valid_count) noexcept -> Tuple<Shared, Other&> {
  iter_count++;
  bool valid = true;
  unrolled_for<iterator_traits_t<Shared>::dimensions>([&](auto dim) {
    if (thread_idx(dim) >= shared.size(dim)) {
      valid = false;
    } else {
      shared.shift(dim, thread_idx(dim) + shared.padding());
    }
  });

  if (valid) {
    *shared = other;
    valid_count++;
  }
  return Tuple<Shared, Other&>{shared, other};
}

/**
 * Offsets the global iterator and ignores the void type. If thee iterator is
 * valid, then valid_count is increased, otherwise it is not.
 *
 * \note This overload is only enabled when only the global parameter is an
 *       iterator.
 *
 * \param  global      An iterator over global memory.
 * \param  iter_count  The number of iterators which have been offset.
 * \param  valid_count The number of iterators which are valid.
 * \tparam Void        The type of the void parameter.
 * \tparam Global      The type of the global memory iterator.
 * \return A tuple with the modified iterator.
 */
template <
  typename Void,
  typename Global,
  only_second_iter_enable_t<Void, Global> = 0>
ripple_host_device auto offset_iters(
  Void, Global& global, uint8_t& iter_count, uint8_t& valid_count) noexcept
  -> Tuple<Global&, Void> {
  iter_count++;
  bool valid = true;
  unrolled_for<iterator_traits_t<Global>::dimensions>([&](auto dim) {
    if (!global.is_valid(dim)) {
      valid = false;
    } else {
      global.shift(dim, global_idx(dim));
    }
  });

  if (valid) {
    valid_count += 1;
  }
  return Tuple<Global&, Void>{global, Void()};
}

/**
 * Forwards back a reference to the other type.
 *
 * \note This overload is only enabled when neither paramters are iterators.
 *
 * \param  other       The type to forward.
 * \param  iter_count  The number of iterators which have been offset.
 * \param  valid_count The number of iterators which are valid.
 * \tparam Void        The type of the void parameter.
 * \tparam Other       The type of the parameter to forward.
 * \return A tuple with the forwarded parameter.
 */
template <
  typename Void,
  typename Other,
  neither_iters_enable_t<Void, Other> = 0>
ripple_host_device auto offset_iters(
  Void, Other&& other, uint8_t& iter_count, uint8_t& valid_count) noexcept
  -> Tuple<Other&, Void> {
  return Tuple<Other&, Void>{other, Void()};
}

/*==--- [shared memory copying] --------------------------------------------==*/

/**
 * Copies the data from the one iterator to the other.
 *
 * \note This is only enabled if both types are iterators.
 *
 * \param  from The iterator to copy from.
 * \param  to   The iterator to copy to.
 * \tparam From The type of the from iterator.
 * \tparam To   The type of the to iterator.
 */
template <typename From, typename To, both_iters_enable_t<From, To> = 0>
ripple_device_only auto
copy_iterator_data(From&& from, To&& to) noexcept -> void {
  *to = *from;
}

/**
 * Overload of copying data for the case that not both types are iterators,
 * and therefore no copy must take place.
 *
 * \note This is a no-op, and it just used for overload resolution.
 *
 * \param  from The iterator to copy from.
 * \param  to   The iterator to copy to.
 * \tparam From The type of the from iterator.
 * \tparam To   The type of the to iterator.
 *
 */
template <typename From, typename To, not_both_iters_enable_t<From, To> = 0>
ripple_device_only auto
copy_iterator_data(From&& from, To&& to) noexcept -> void {}

/*==--- [invoke implementation] --------------------------------------------==*/

/**
 * Actually executes the invocable witht the given arguments.
 *
 * \param  invocable       The invocable to execute.
 * \param  num_iters       The number of iterators.
 * \param  num_valid_iters The number of valid iterators.
 * \param  args            The arguments for the invocable.
 * \tparam Invocable       The type of the invocable.
 * \tparam Args            The types of the arguments.
 */
template <typename Invocable, typename... Args>
ripple_device_only auto execute_invocable(
  Invocable&& invocable,
  uint8_t&    iters,
  uint8_t&    valid,
  Args&&... args) noexcept -> void {
  // Sync needed here because of shared to global load:
  if (valid > 0) {
    sync_block();
  }
  invocable(get<0>(args)...);

  // For any argument which are shared memory iterators, we need to copy
  // the results back from shared memory to global memory:
  (copy_iterator_data(get<0>(args), get<1>(args)), ...);
}

/**
 * Expands the arguments and wrapped arguments into the invocable so that it
 * can be executed.
 *
 * \param  invocable    The invocable to execute.
 * \param  data         A pointer to the shared memory buffer.
 * \param  wrapped_args Wrapped arguments which hold shared memory
 *                      information.
 * \param  args         Original arguments for the invocable.
 * \tparam Invocable    The invocable to execute.
 * \tparam Ts           The types of the wrapped arguments.
 * \tparam Args         The type of the arguments.
 * \tparam I            The indices of the arguments to expand.
 */
template <typename Invocable, typename... Ts, typename... Args, size_t... I>
ripple_device_only auto expand_into_invocable(
  Invocable&&   invocable,
  void*         data,
  Tuple<Ts...>& wrapped_args,
  std::index_sequence<I...>,
  Args&&... args) noexcept -> void {
  // Count total number of iterators, and how many are valid.
  uint8_t iter_count = 0, valid_count = 0;

  // Here, offset_iters and make_shared_iterator exctract any iterators over
  // shared memory and offset them appropriately, copy data between global
  // and shared where necessary, and synchronize those operations.
  execute_invocable(
    static_cast<Invocable&&>(invocable),
    iter_count,
    valid_count,
    offset_iters(
      make_shared_iterator(get<I>(wrapped_args), data),
      static_cast<Args&&>(args),
      iter_count,
      valid_count)...);
}

/**
 * Global kernel which does the actual invoking, expanding the arguments into
 * the invocable.
 *
 * This also creates a shared memory buffer for the case that shared memory is
 * used.
 *
 * \param  invocable    The invocable to execute.
 * \param  wrapped_args Wrapped arguments which hold shared memory
 * information. \param  args         Original arguments for the invocable.
 * \tparam Invocable    The invocable to execute.
 * \tparam Ts           The types of the wrapped arguments.
 * \tparam Args         The type of the arguments.
 */
template <typename Invocable, typename... Ts, typename... Args>
ripple_global auto invoke_impl(
  Invocable invocable, Tuple<Ts...> wrapped_args, Args... args) noexcept
  -> void {
  extern __shared__ char buffer[];
  expand_into_invocable(
    invocable,
    static_cast<void*>(buffer),
    wrapped_args,
    std::make_index_sequence<sizeof...(Ts)>(),
    args...);
}

/**
 * Implementation of generic invoke for the gpu.
 *
 * This will look at the types of the arguments, and for any which are Block
 * types, or BlockEnabled, will pass them as offset iterators to the
 * invocable.
 *
 * If any of the arguments are wrapped with shared wrapper types, they are
 * passed as iterators over shared memory.
 *
 * \note If this uses a stream from one of the arguments, the stream *is not*
 *       synchronized. However, if a stream need to be created, then that
 *       stream is synchronized.
 *
 * \param  invocable The invocable to execute on the gpu.
 * \param  args      The arguments for the invocable.
 * \tparam Invocable The type of the invocable.
 * \tparam Args      The type of the args.
 */
template <typename Invocable, typename... Args>
auto invoke_generic_impl(Invocable&& invocable, Args&&... args) noexcept
  -> void {
#if defined(__CUDACC__)
  constexpr size_t dims =
    max_element(block_enabled_traits_t<Args>::dimensions...);

  // Find the grid size:
  const auto sizes = std::array<size_t, 3>{
    max_element(size_t{1}, get_block_size(args, dim_x)...),
    max_element(size_t{1}, get_block_size(args, dim_y)...),
    max_element(size_t{1}, get_block_size(args, dim_z)...)};

  // Gets the size of the grid to run on the gpu. Currently this only uses
  // dynamic parameters, because the performance difference is minimal, but
  // we should add the options for statically sizes parameters.
  const auto exec_params = dynamic_params<dims>();
  auto [threads, blocks] = get_execution_sizes(exec_params, sizes);

  // Create a stream, incase we dont have one to run on. Also get any other
  // types which may have been requested to be passed in shared memory, and
  // compute the amount of shared memory.
  Stream stream;
  size_t shared_mem = 0;
  auto   all_params = make_tuple(
    create_shared_info<dims>(shared_mem, stream, exec_params, args)...);

  // If there are no streams, then we need to create one.
  if (!stream.set) {
    cudaStreamCreate(&stream.stream);
  }

  // Run the kernel. For any parameters which are Block types, we convert the
  // block to an iterator over the block.
  invoke_impl<<<blocks, threads, shared_mem, stream.stream>>>(
    invocable, all_params, util::block_iter_or_same(args)...);

  if (!stream.set) {
    cudaStreamSynchronize(stream.stream);
    cudaStreamDestroy(stream.stream);
  }
#endif
}

} // namespace ripple::kernel::gpu

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_PIPELINE_GENERIC_CUDA__HPP
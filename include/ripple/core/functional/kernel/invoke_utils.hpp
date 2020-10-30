//==--- ripple/core/functional/kernel/invoke_utils_.hpp ---- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  invoke_utils_.hpp
/// \brief This file implements functionality to invoke a pipeline on various
///        container objects on the device.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP
#define RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP

#include <ripple/core/container/shared_wrapper.hpp>
#include <ripple/core/execution/execution_size.hpp>

namespace ripple::kernel {

/** Alias for dimension sizes. */
using DimSizes = std::array<size_t, 3>;

//==--- [block extraction] -------------------------------------------------==//

template <typename T>
auto extract_device_block(SharedWrapper<T> wrapper) -> SharedWrapper<T> {
  return SharedWrapper<T>{wrapper.wrapped, wrapper.padding};
}

template <typename T>
decltype(auto) extract_device_block(T&& t) {
  return t;
}

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
auto get_size_if_block(T&& block, Dim&& dim) noexcept -> size_t {
  return block_enabled_traits_t<T>::dimensions > dim
           ? block.size(static_cast<Dim>(dim))
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
auto get_size_if_block(T&& block, Dim&& dim) noexcept -> size_t {
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
auto get_size_if_block(SharedWrapper<T>& wrapper, Dim&& dim) noexcept
  -> size_t {
  return get_size_if_block(wrapper.wrapped, static_cast<Dim&&>(dim));
}

namespace gpu::util {

/**
 * Returns an iterator over the block data if the parameter is block enabled.
 *
 * \note We need the decay here to ensure that we don't get a reference, which
 *       would be undefined if passed to the device.
 *
 * \param  block The block enabled type to get an iterator for.
 * \tparam T     The type of the block enabled type.
 * \return An iterator over the device data.
 */
template <typename T, block_enabled_t<T> = 0>
auto block_iter_or_same(T&& block) noexcept
  -> std::decay_t<decltype(block.device_iterator())> {
  return block.device_iterator();
}

/**
 * Gets a decated  instance of the parameter.
 *
 * \note We need the decay here to ensure that we don't get a reference, which
 *       would be undefined if passed to the device.
 *
 * \param  t The block enabled type to get an iterator for.
 * \tparam T The type of the block enabled type.
 * \return A decayed instance of the parameter.
 */
template <typename T, non_block_enabled_t<T> = 0>
auto block_iter_or_same(T&& t) noexcept -> std::decay_t<T> {
  return t;
}

/**
 * Returns an iterator over the block data if the parameter being wrapped is a
 * block type.
 *
 * \param  wrapper The block enabled wrapper type.
 * \tparam T       The type of the block enabled type.
 * \return An iterator over the wrapped data.
 */
template <typename T, block_enabled_t<T> = 0>
auto block_iter_or_same(SharedWrapper<T>& wrapper) noexcept
  -> std::decay_t<decltype(wrapper.wrapped.device_iterator())> {
  return wrapper.wrapped.device_iterator();
}

/**
 * Gets a decayed instance of the wrapped type.
 *
 * \param  wrapper The non block enabled wrapper type.
 * \tparam T       The type of the non block enabled type.
 * \return A decayed instance of the wrapped type.
 */
template <typename T, non_block_enabled_t<T> = 0>
auto block_iter_or_same(SharedWrapper<T>& wrapper) noexcept -> std::decay_t<T> {
  return wrapper.wrapped;
}

/**
 * Returns an iterator over the block data if the parameter being wrapped is a
 * block type.
 *
 * \param  wrapper The block enabled wrapper type.
 * \tparam T       The type of the block enabled type.
 * \return An iterator over the wrapped data.
 */
template <typename T, block_enabled_t<T> = 0>
auto block_iter_or_same(const SharedWrapper<T>& wrapper) noexcept
  -> std::decay_t<decltype(wrapper.wrapped.device_iterator())> {
  return wrapper.wrapped.device_iterator();
}

/**
 * Gets a decayed instance of the wrapped type.
 *
 * \param  wrapper The non block enabled wrapper type.
 * \tparam T       The type of the non block enabled type.
 * \return A decayed instance of the wrapped type.
 */
template <typename T, non_block_enabled_t<T> = 0>
auto block_iter_or_same(const SharedWrapper<T>& wrapper) noexcept
  -> std::decay_t<T> {
  return wrapper.wrapped;
}

} // namespace gpu::util

} // namespace ripple::kernel

#endif // RIPPLE_FUNCTIONAL_KERNEL_INVOKE_UTILS__HPP
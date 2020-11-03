//==--- ripple/core/container/block_extractor.hpp ---------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block_extractor.hpp
/// \brief This file implements functionality for extracting blocks from a ///
///        tensor.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_BLOCK_EXTRACTOR_HPP
#define RIPPLE_CONTAINER_BLOCK_EXTRACTOR_HPP

#include "shared_wrapper.hpp"
#include "tensor_traits.hpp"
#include <ripple/core/graph/modifier.hpp>

namespace ripple {

/**
 * This struct extracts the blocks from a tensor.
 */
struct BlockExtractor {
  /**
   * Extracts the blocks from a tensor.
   *
   * \note This overload is for non tensor types and simply forward the given
   *       type.
   *
   * \param  t The object to extract the blocks from.
   * \tparam T The type of the object.
   * \return An rvalue rerence to the object.
   */
  template <typename T, non_tensor_enable_t<T> = 0>
  static auto extract_blocks_if_tensor(T&& t) noexcept -> T&& {
    return ripple_forward(t);
  }

  /**
   * Extracts the blocks from a tensor.
   *
   * \note This overload is for tensor types and returns a reference to the
   *       blocks which define the tensor.
   *
   * \param  t The tensor object to extract the blocks from.
   * \tparam T The type of the tensor object.
   * \return An reference to the tensor blocks.
   */
  template <typename T, tensor_enable_t<T> = 0>
  static auto extract_blocks_if_tensor(T& t) noexcept
    -> std::remove_reference_t<decltype(t.blocks_)>& {
    return t.blocks_;
  }

  /**
   * Extracts the blocks from a tensor.
   *
   * \note This overload is for tensor types wrapped in a modification
   *       specifier and returns a reference to blocks, rewrapped in the
   *       specifier.
   *
   * \param   specifier The modification specifier to extract the blocks from.
   * \tparam  T         The type of the tensor object.
   * \tparmam M         The type of the modification.
   * \return A new modification specifier which references the tensor blocks.
   */
  template <typename T, Modifier M, tensor_enable_t<T> = 0>
  static auto
  extract_blocks_if_tensor(ModificationSpecifier<T, M> specifier) noexcept {
    using Spec = ModificationSpecifier<decltype(specifier.wrapped.blocks_)&, M>;
    return Spec{specifier.wrapped.blocks_};
  }

  /**
   * Extracts the blocks from a tensor.
   *
   * \note This overload is for tensor types wrapped in a shared wrapper
   *       and returns a reference to blocks, rewrapped in the wrapper.
   *       specifier.
   *
   * \param   wrapper The shared wrapper to extract the blocks from.
   * \tparam  T       The type of the tensor object.
   * \return A new shared wrapper which references the tensor blocks.
   */
  template <typename T, tensor_enable_t<T> = 0>
  static auto extract_blocks_if_tensor(SharedWrapper<T> wrapper) noexcept {
    using Wrapper = SharedWrapper<decltype(wrapper.wrapped.blocks_)&>;
    return Wrapper{wrapper.wrapped.blocks_};
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BLOCK_EXTRACTOR_HPP

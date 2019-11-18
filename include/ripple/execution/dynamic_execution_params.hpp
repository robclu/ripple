//==--- ripple/execution/dynamic_execution_params.hpp ------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dynamic_execution_params.hpp
/// \brief This file implements dynamic execution parameters.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_DYNAMIC_EXECUTION_PARAMS_HPP
#define RIPPLE_EXECUTION_DYNAMIC_EXECUTION_PARAMS_HPP

#include "execution_params.hpp"
#include <ripple/iterator/block_iterator.hpp>
#include <ripple/multidim/dynamic_multidim_space.hpp>
#include <ripple/storage/storage_traits.hpp>

namespace ripple {

/// The DynamicExecParams struct defines the parameters of an execution space
/// for which the size is dynamic.
///
/// \tparam Grain   The number of elements to processes per thread in the tile.
/// \tparam Shared  A type for tile local shared memory.
template <std::size_t Grain, typename Shared>
struct DynamicExecParams : public ExecParams<DynamicExecParams<Grain, Shared>> {
 private:
  //==--- [aliases] --------------------------------------------------------==//
 
  /// Defines the layout traits for the shared memory type.
  using traits_t    = layout_traits_t<Shared>;
  /// Defines the value type for the iterator over the execution space.
  using value_t     = typename traits_t::value_t;
  /// Defines the allocator type for the execution space.
  using allocator_t = typename traits_t::allocator_t;
  /// Defines the type of the multidimensional space for the execution.
  using space_t     = DynamicMultidimSpace<3>;
  /// Defines the type of the iterator over the execution space.
  using iter_t      = BlockIterator<value_t, space_t>;

 public:
  //==--- [constructor] ----------------------------------------------------==//
  
  /// Creates the execution space without padding.
  /// \tparam Sizes The sizes of the execution space.
  template <typename... Sizes, all_arithmetic_size_enable_t<3, Sizes...> = 0>
  ripple_host_device constexpr DynamicExecParams(Sizes&&... sizes)
  : _space{static_cast<std::size_t>(sizes)...} {}

  /// Creates the eecution space without padding.
  /// \tparam Sizes The sizes of the execution space.
  template <typename... Sizes, all_arithmetic_size_enable_t<3, Sizes...> = 0>
  ripple_host_device constexpr 
  DynamicExecParams(std::size_t padding, Sizes&&... sizes)
  : _space{padding, static_cast<std::size_t>(sizes)...} {}
  
  //==--- [size] -----------------------------------------------------------==//
  
  /// Returns the total size of the execution space, for Dims dimensions.
  /// \tparam Dims The number of dimensions to get the size for.
  template <std::size_t Dims>
  ripple_host_device constexpr auto size() const -> std::size_t {
    static_assert(
      Dims <= 3, "Execution space can't be more than 3 dimensions!"
    );
    auto total_size = _space.size(dim_x);
    unrolled_for<Dims - 1>([&] (auto d) {
      constexpr auto dim = d + 1;
      total_size *= _space.size(dim);
    });
    return total_size;
  }

  /// Returns the size of the space in the \p dim dimension, without the
  /// padding.
  /// \tparam dim The dimension to get the size of the space for.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> std::size_t {
    return _space.internal_size(std::forward<Dim>(dim));
  }

  //==--- [properties] -----------------------------------------------------==//

  /// Returns that the space is not of a fixed size.
  ripple_host_device constexpr auto is_fixed() const -> bool {
    return false;
  }

  /// Returns the grain size for the space (the number of elements to be
  /// processed by a thread in the space).
  ripple_host_device constexpr auto grain_size() const -> std::size_t {
    return Grain;
  }

  /// Returns the grain index for the space, the current element being processed
  /// in the space.
  ripple_host_device constexpr auto grain_index() -> std::size_t& {
    return _grain_index;
  }

  /// Returns the amount of padding for the execution space.
  ripple_host_device constexpr auto padding() const -> std::size_t {
    return _space.padding();
  }

  //==--- [creation] -------------------------------------------------------==//

  /// Returns an iterator over a memory space pointed to by \p data.
  /// \param data A pointer to the memory space data.
  template <typename T>
  ripple_host_device auto iterator(T* data) const -> iter_t {
    return iter_t{allocator_t::create(data, _space), _space};
  }

  /// Returns the number of bytes required to allocator data for the space.
  /// \tparam Dims The number of dimensions to allocate for.
  template <std::size_t Dims>
  ripple_host_device constexpr auto allocation_size() const -> std::size_t {
    return allocator_t::allocation_size(size<Dims>());
  }
 private:
  space_t     _space;            //!< The space defining the execution region.
  std::size_t _grain_index = 0;  //!< The index of the grain element.
};

} // namespace ripple

#endif // RIPPLE_EXECUTION_DYNAMIC_EXECUTION_PARAMS_HPP


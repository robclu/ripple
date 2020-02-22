//==--- ripple/core/execution/dynamic_execution_params.hpp ------ -*- C++ -*- ---==//
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
#include <ripple/core/iterator/block_iterator.hpp>
#include <ripple/core/multidim/dynamic_multidim_space.hpp>
#include <ripple/core/storage/storage_traits.hpp>

namespace ripple {

/// The DynamicExecParams struct defines the parameters of an execution space
/// for which the size is dynamic.
///
/// \tparam Shared  A type for tile local shared memory.
template <typename Shared>
struct DynamicExecParams : public ExecParams<DynamicExecParams<Shared>> {
 private:
  //==--- [aliases] --------------------------------------------------------==//
 
  /// Defines the layout traits for the shared memory type.
  using traits_t    = layout_traits_t<Shared>;
  /// Defines the value type for the iterator over the execution space.
  using value_t     = typename traits_t::value_t;
  /// Defines the allocator type for the execution space.
  using allocator_t = typename traits_t::allocator_t;
  /// Defines the type of the default space for the execution.
  using space_t     = DynamicMultidimSpace<3>;

  /// Defines the type of the multidimensional space for the execution.
  /// \tparam Dims The number of dimensions for the space.
  template <size_t Dims>
  using make_space_t = DynamicMultidimSpace<Dims>;

 public:
  //==--- [constructor] ----------------------------------------------------==//
 
  /// Default constructor, creates a space of size 1024, 1, 1 with no padding.
  ripple_host_device constexpr DynamicExecParams()
  : _space{1024, 1, 1} {}

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

  /// Returns the amount of padding for the execution space.
  ripple_host_device constexpr auto padding() const -> std::size_t {
    return _space.padding();
  }

  /// Returns the target architecture for the computation.
  ripple_host_device constexpr auto target_arch() const -> ComputeArch {
    return _arch;
  }

  /// Returns a reference to the target architecture for the computation.
  ripple_host_device constexpr auto target_arch() -> ComputeArch& {
    return _arch;
  }

  //==--- [creation] -------------------------------------------------------==//

  /// Returns an iterator over a memory space pointed to by \p data.
  /// \param data A pointer to the memory space data.
  template <size_t Dims, typename T>
  ripple_host_device auto iterator(T* data) const {
    using _space_t = make_space_t<Dims>;
    using iter_t   = BlockIterator<value_t, _space_t>;
    _space_t space;
    unrolled_for<Dims>([&] (auto d) {
      constexpr auto dim = d;
      space[dim] = _space[dim];
    });
    space.padding() = _space.padding();
    return iter_t{allocator_t::create(data, space), space};
  }

  /// Returns the number of bytes required to allocator data for the space.
  /// \tparam Dims The number of dimensions to allocate for.
  template <std::size_t Dims>
  ripple_host_device constexpr auto allocation_size() const -> std::size_t {
    return allocator_t::allocation_size(size<Dims>());
  }
 private:
  space_t     _space;                       //!< The execution space.
  ComputeArch _arch = ComputeArch::device;  //!< The arch for the computation.
};

//==--- [functions] --------------------------------------------------------==//

/// Creates default dynamic execution execution paramters for the device.
/// \tparam Dims    The number of dimensions for the parameters.
/// \tparam Shared  The type of the shared memory for the parameters.
template <size_t Dims, typename Shared = VoidShared>
ripple_host_device auto dynamic_device_params() 
-> DynamicExecParams<Shared> {
  constexpr auto size_x = (Dims == 1 ? 1024 : Dims == 2 ? 32 : 8);
  constexpr auto size_y = (Dims == 1 ? 1    : Dims == 2 ? 16 : 8);
  constexpr auto size_z = (Dims == 1 ? 1    : Dims == 2 ? 1  : 8);

  return DynamicExecParams<Shared>(size_x, size_y, size_z);
}

/// Creates default dynamic execution execution paramters for the host.
/// \tparam Dims    The number of dimensions for the parameters.
/// \tparam Shared  The type of the shared memory for the parameters.
template <size_t Dims, typename Shared = VoidShared>
ripple_host_device auto dynamic_host_params() 
-> DynamicExecParams<Shared> {
  constexpr auto size_x = (Dims == 1 ? 1024 : Dims == 2 ? 32 : 8);
  constexpr auto size_y = (Dims == 1 ? 1    : Dims == 2 ? 16 : 8);
  constexpr auto size_z = (Dims == 1 ? 1    : Dims == 2 ? 1  : 8);

  auto params = DynamicExecParams<Shared>(size_x, size_y, size_z);
  params.target_arch() = ComputeArch::host;
  return params;
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_DYNAMIC_EXECUTION_PARAMS_HPP


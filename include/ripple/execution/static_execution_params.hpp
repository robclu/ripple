//==--- ripple/execution/static_execution_params.hpp -------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  static_execution_params.hpp
/// \brief This file implements compile time execution parameters.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_STATIC_EXECUTION_PARAMS_HPP
#define RIPPLE_EXECUTION_STATIC_EXECUTION_PARAMS_HPP

#include "execution_params.hpp"
#include <ripple/iterator/block_iterator.hpp>
#include <ripple/multidim/static_multidim_space.hpp>
#include <ripple/storage/storage_traits.hpp>

namespace ripple {

/// The StaticExecParams struct defines the parameters of an execution space for
/// which the size is static, and known at compile time.
///
/// \tparam SizeX   The number of threads to execute in the x dimension.
/// \tparam SizeY   The number of threads to execute in the y dimension..
/// \tparam SizeZ   The number of threads to execute in the z dimension.
/// \tparam Padding The amount of padding for each size of each dimension.
/// \tparam Shared  A type for tile local shared memory.
template <
  std::size_t SizeX  ,
  std::size_t SizeY  ,
  std::size_t SizeZ  ,
  std::size_t Padding,
  typename    Shared
>
struct StaticExecParams : public 
  ExecParams<StaticExecParams<SizeX, SizeY, SizeZ, Padding, Shared>> {
 private:
  //==--- [aliases] --------------------------------------------------------==//
 
  /// Defines the layout traits for the shared memory type.
  using traits_t    = layout_traits_t<Shared>;
  /// Defines the value type for the iterator over the execution space.
  using value_t     = typename traits_t::value_t;
  /// Defines the allocator type for the execution space.
  using allocator_t = typename traits_t::allocator_t;

  /// Defines the type of the multidimensional space for the execution.
  /// \tparam Dims The number of dimensions to get the space for.
  template <size_t Dims>
  using space_t =
    std::conditional_t<
      Dims == 1, StaticMultidimSpace<SizeX>,
      std::conditional_t<
        Dims == 2, 
        StaticMultidimSpace<SizeX, SizeY>,
        StaticMultidimSpace<SizeX, SizeY, SizeZ>
      >
    >;

  /// Defines the type of the iterator over the execution space.
  /// \tparam Dims The number of dimensions for the iterator.
  template <std::size_t Dims>
  using iter_t = BlockIterator<value_t, space_t<Dims>>;

 public:
  //==--- [size] -----------------------------------------------------------==//
  
  /// Returns the total size of the execution space, for Dims dimensions.
  /// \tparam Dims The number of dimensions to get the size for.
  template <std::size_t Dims>
  ripple_host_device constexpr auto size() const -> std::size_t {
    constexpr auto dim_pad     = Padding * 2;
    constexpr auto dim_1d_size = SizeX + dim_pad;
    constexpr auto dim_2d_size = dim_1d_size * (SizeY + dim_pad);
    constexpr auto dim_3d_size = dim_2d_size * (SizeZ + dim_pad);
    return 
      Dims == 1 ? dim_1d_size :
      Dims == 2 ? dim_2d_size :
      Dims == 3 ? dim_3d_size : 0;
  }

  /// Returns the size of the space in the \p dim dimension.
  /// \tparam dim The dimension to get the size of the space for.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> std::size_t {
    return size_impl(std::forward<Dim>(dim));
  }

  //==--- [properties] -----------------------------------------------------==//

  /// Returns the amount of padding for the execution space.
  ripple_host_device constexpr auto padding() const -> std::size_t {
    return Padding;
  }

  //==--- [creation] -------------------------------------------------------==//

  /// Returns an iterator over a memory space pointed to by \p data, for a
  /// specific number of dimension.
  /// \param  data A pointer to the memory space data.
  /// \tparam Dims The number of dimensions for the iterator.
  /// \tparam T    The type of the data.
  template <std::size_t Dims, typename T>
  ripple_host_device auto iterator(T* data) const {
    using _space_t       = space_t<Dims>;
    using _iter_t        = BlockIterator<value_t, _space_t>;
    constexpr auto space = _space_t{Padding};
    return _iter_t{allocator_t::create(data, space), space};
  }

  /// Returns the number of bytes required to allocator data for the space.
  /// \tparam Dims The number of dimensions to allocate for.
  template <std::size_t Dims>
  ripple_host_device constexpr auto allocation_size() const -> std::size_t {
    return allocator_t::allocation_size(size<Dims>());
  }

 private:
  //==--- [methods] --------------------------------------------------------==//
  
  /// Implementation to return the size of the execution space in the x
  /// dimension.
  ripple_host_device constexpr auto size_impl(dimx_t) const -> std::size_t {
    return SizeX;
  }

  /// Implementation to return the size of the execution space in the y
  /// dimension.
  ripple_host_device constexpr auto size_impl(dimy_t) const -> std::size_t {
    return SizeY;
  }

  /// Implementation to return the size of the execution space in the z
  /// dimension.
  ripple_host_device constexpr auto size_impl(dimz_t) const -> std::size_t {
    return SizeZ;
  }
};

} // namespace ripple

#endif // RIPPLE_EXECUTION_STATIC_EXECUTION_PARAMS_HPP

//==--- ripple/execution/execution_params.hpp -------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  execution_params.hpp
/// \brief This file contains functionality for defining parameters for
///        execution.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP
#define RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP

#include <ripple/utility/dim.hpp>
#include <ripple/utility/type_traits.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/// The ExecParams struct defines the parameters of the space for execution.
/// \tparam SizeX  The number of threads to execute in the x dimension.
/// \tparam SizeY  The number of threads to execute in the y dimension..
/// \tparam SizeZ  The number of threads to execute in the z dimension.
/// \tparam Gran   The number of elements to processes per thread in the tile.
/// \tparam Shared A type for tile local shared memory.
template <
  std::size_t SizeX,
  std::size_t SizeY  = 1,
  std::size_t SizeZ  = 1,
  std::size_t Grain  = 1,
  typename    Shared = std::void_t<>
>
struct ExecParams {
  ///==--- [constants] -----------------------------------------------------==//

  /// Defines the total size of the tile.
  static constexpr auto total_size = SizeX * SizeY * SizeZ;
  /// Defines the grain size of the tile.
  static constexpr auto grain      = Grain;

  //==--- [interface] ------------------------------------------------------==//

  /// Returns the size of the tile in the \p dim dimension.
  /// \tparam dim The dimension to get the size of the tile for.
  template <typename Dim>
  ripple_host_device static constexpr auto size(Dim&& dim) -> std::size_t {
    return size_impl(std::forward<Dim>(dim));
  }

  std::size_t grain_index = 0;  //!< The index of the grain element.

 private:
  /// Implementation to return the size of the execution space in the x
  /// dimension.
  ripple_host_device static constexpr auto size_impl(dimx_t) -> std::size_t {
    return SizeX;
  }

  /// Implementation to return the size of the execution space in the y
  /// dimension.
  ripple_host_device static constexpr auto size_impl(dimy_t) -> std::size_t {
    return SizeY;
  }

  /// Implementation to return the size of the execution space in the z
  /// dimension.
  ripple_host_device static constexpr auto size_impl(dimz_t) -> std::size_t {
    return SizeZ;
  }
};

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTION_PARAMS_HPP

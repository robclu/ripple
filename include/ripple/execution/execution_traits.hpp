//==--- ripple/execution/execution_traits.hpp -------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  execution_traits.hpp
/// \brief This file defines traits for execution.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EXECUTION_EXECUTION_TRAITS_HPP
#define RIPPLE_EXECUTION_EXECUTION_TRAITS_HPP

#include <ripple/utility/portability.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// Defines a void type for a void shared memory execution parameter.
struct VoidShared {};

/// The ExecParams struct defines an interface for all execution parameter
/// implementations.
/// \tparam Impl The implementation of the interface.
template <typename Impl> struct ExecParams;

/// The StaticExecParams struct defines the parameters of an execution space for
/// which the size is static, and known at compile time.
///
/// \tparam SizeX   The number of threads to execute in the x dimension.
/// \tparam SizeY   The number of threads to execute in the y dimension..
/// \tparam SizeZ   The number of threads to execute in the z dimension.
/// \tparam Grain   The number of elements to processes per thread in the tile.
/// \tparam Padding The amount of padding for each size of each dimension.
/// \tparam Shared  A type for tile local shared memory.
template <
  std::size_t SizeX      ,
  std::size_t SizeY   = 1,
  std::size_t SizeZ   = 1,
  std::size_t Grain   = 1,
  std::size_t Padding = 0,
  typename    Shared  = VoidShared
>
struct StaticExecParams;

/// The DynamicExecParams struct defines the parameters of an execution space
/// for which the size is dynamic.
///
/// \tparam Grain   The number of elements to processes per thread in the tile.
/// \tparam Shared  A type for tile local shared memory.
template <std::size_t Grain = 1, typename Shared = VoidShared>
struct DynamicExecParams;

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type \t implements the ExecParams interface.
/// \tparam T The type to check if implements the exec params interface.
template <typename T>
static constexpr auto is_exec_params_v = 
  std::is_base_of_v<ExecParams<std::decay_t<T>>, std::decay_t<T>>;

//==--- [aliases] ----------------------------------------------------------==//

/// Defines an alias for 1d execution parameters. 1024 threads in the x
/// dimension, a grain size of 1, and no padding.
using exec_params_1d_t = StaticExecParams<1024>;

/// Defines an alias for 2d execution parameters. 32 threads in the x
/// dimension, 16 in the y dimension, a grain size of 1, and no padding.
using exec_params_2d_t = StaticExecParams<32, 16>;

/// Defines an alias for 3d execution parameters. 8 threads in the x dimension,
/// 8 threads, in the y dimension, 8 threads in the z dimension, a grain size
/// of 1, and no padding.
using exec_params_3d_t = StaticExecParams<8, 8, 8>;

/// Defines the execution parameter type based on the number of dimensions.
/// \tparam Dims The number of dimensions to get the execution params for.
template <std::size_t Dims>
using default_exec_params_t = 
  std::conditional_t<Dims == 1, exec_params_1d_t,
    std::conditional_t<Dims == 2, exec_params_2d_t, exec_params_3d_t>
  >;

//==--- [enables] ----------------------------------------------------------==//

/// Defines a valid type if the grain size is one.
/// \tparam ExecImpl The execution params implementation to base the enable on.
template <typename ExecImpl, typename Exec = std::decay_t<ExecImpl>>
using single_grain_enable_t = std::enable_if_t<
  (!is_exec_params_v<Exec> || std::declval<Exec>().grain_size() == 1), int
>;

/// Defines a valid type if the grain size is greater than one.
/// \tparam ExecImpl The execution params implementation to base the enable on.
template <typename ExecImpl, typename Exec = std::decay_t<ExecImpl>>
using multi_grain_enable_t = std::enable_if_t<
  (is_exec_params_v<Exec> && std::declval<Exec>().grain_size() > 1), int
>;

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTION_TRAITS_HPP

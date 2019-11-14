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

#include "execution_params.hpp"

namespace ripple {

/// Defines an alias for 1d execution parameters. 1024 threads in the x
/// dimension, and a grain size of 1.
using exec_params_1d_t = ExecParams<1024, 1, 1, 1>;

/// Defines an alias for 2d execution parameters. 32 threads in the x dimension,
/// 16 in the y dimension, and a grain size of 1.
using exec_params_2d_t = ExecParams<32, 16, 1, 1>;

/// Defines an alias for 2d execution parameters. 8 threads in the x dimension,
/// 8 in the y dimension, 8 threads in the z dimension, and a grain size of 1.
using exec_params_3d_t = ExecParams<8, 8, 8, 1>;

/// Defines the execution parameter type based on the number of dimensions.
/// \tparam Dims The number of dimensions to get the execution params for.
template <std::size_t Dims>
using default_exec_params_t = 
  std::conditional_t<Dims == 1, exec_params_1d_t,
    std::conditional_t<Dims == 2, exec_params_2d_t, exec_params_3d_t>
  >;

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTION_TRAITS_HPP

/**==--- ripple/core/utility/forward.hpp ------------------- -*- C++ -*- ---==**
 *
 *                                Ripple
 *
 *                  Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==------------------------------------------------------------------------==**
 *
 * \file  forward.hpp
 * \brief This file defines macros for forwarding and moving.
 *
 *==------------------------------------------------------------------------==*/

/*
 * std::forward and std::move require utility, which along with the need to do
 * name lookup, overload resolution, and template instantiation, and the compile
 * time effects are significant.
 *
 * When modules are supported by nvcc, we can remove these macros and use
 * std::forward and std::move.
 */

#ifndef RIPPLE_UTILITY_FORWARD_HPP
#define RIPPLE_UTILITY_FORWARD_HPP

#include <type_traits>

/** Static case to rvalue refrence. */
#define ripple_move(...) \
  static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)

/** Static cast to identity. Don't need the &&, but it's more robust. */
#define ripple_forward(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)

#endif // RIPPLE_UTILITY_FORWARD_HPP
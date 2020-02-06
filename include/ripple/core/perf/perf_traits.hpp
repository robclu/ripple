//==--- ripple/core/perf/perf_traits.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Ripple.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  perf_traits.hpp
/// \brief This file implements traits related to the perf module.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_PERF_PERF_TRAITS_HPP
#define RIPPLE_PERF_PERF_TRAITS_HPP

#include "event.hpp"
#include <ripple/core/utility/type_traits.hpp>

namespace ripple {

/// Returns true if the type T is an event.
/// \tparam T The type to determine if is an event.
template <typename T>
static constexpr auto is_event_v = std::is_same_v<Event, std::decay_t<T>>;

/// Defines a valid type if the type T is an Event.
/// \tparam T The type to define the enable on.
template <typename T>
using event_enable_t = std::enable_if_t<is_event_v<std::decay_t<T>>, int>;

/// Defines a valid type if the type T is not an Event.
/// \tparam T The type to define the enable on.
template <typename T>
using non_event_enable_t = std::enable_if_t<!is_event_v<std::decay_t<T>>, int>;

} // namespace ripple

#endif // RIPPLE_PERF_PERF_TRAITS_HPP

//==--- ripple/perf/event.hpp ------------------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  event.hpp
/// \brief This file defines a class to represent an event.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_PERF_EVENT_HPP
#define RIPPLE_PERF_EVENT_HPP

namespace ripple {

/// The Event class stores information pertaining to an event.
struct Event {
  /// The time over which the event elapsed.
  float elapsed_time_ms = 0.0f;
};

} // namespace ripple

#endif // RIPPLE_PERF_EVENT_HPP

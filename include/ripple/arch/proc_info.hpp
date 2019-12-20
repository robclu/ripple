//==--- ripple/arch/proc_info.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  proc_info.hpp
/// \brief This file defines a struct to store processor information.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_PROC_INFO_HPP
#define RIPPLE_ARCH_PROC_INFO_HPP

namespace ripple {

/// The ProcInfo struct stores information about the processor based on where
/// it is in the system topology.
struct ProcInfo {
  /// Defines an invalid value for processor information.
  static constexpr uint32_t invalid = 0xFFFFFFFF;

  /// Returns true if any field in the information is invalid.
  auto is_invalid() const -> bool {
    return package == invalid || core == invalid || thread == invalid;
  }

  uint32_t package = invalid;   //!< Package to which the processor belongs.
  uint32_t core    = invalid;   //!< Index of the core in the package.
  uint32_t thread  = invalid;   //!< The logical thread index on the core.
};

} // namespace ripple

#endif // RIPPLE_ARCH_PROC_INFO_HPP



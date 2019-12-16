//==--- ripple/arch/cache.hpp ------------------------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cache.hpp
/// \brief This file defiens a struct for a cache.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_CACHE_HPP
#define RIPPLE_ARCH_CACHE_HPP

#include <cstdint>

namespace ripple {

/// Stores information for a cache.
struct Cache {
  /// Defines the type of the cache, as per the Intel spec.
  enum Type : uint32_t {
    Null        = 0x0,      //!< Null or invalid cache.
    Data        = 0x1,      //!< Data cache.
    Instruction = 0x2,      //!< Instruction cache
    Unified     = 0x3       //!< Unified cache.
  };

  /// Returns the size of the cache in kB.
  auto size() const -> uint32_t {
    return assosciativity * partitions * linesize * sets / 1024;
  }

  Type     type           = Type::Null; //!> Type of the cache
  uint32_t level          = 0;          //!< Level of the cache
  uint32_t linesize       = 0;          //!< Cache line size in bytes.
  uint32_t partitions     = 0;          //!< Number of partitions.
  uint32_t assosciativity = 0;          //!< Ways of assosciativity.
  uint32_t shared_by      = 0;          //!< Max threads sharing this cache.
  uint32_t sets           = 0;          //!< Number of sets.
  uint32_t mask           = 0;          //!< Mask to determine cache sharing.
};

} // namespace ripple

#endif // RIPPLE_ARCH_CACHE_HPP

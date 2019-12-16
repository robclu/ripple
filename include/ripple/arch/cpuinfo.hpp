//==--- ripple/arch/cpuinfo.hpp ---------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cpuinfo.hpp
/// \brief This file defines functionality for cpu information.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_CPUINFO_HPP
#define RIPPLE_ARCH_CPUINFO_HPP

#include "cache.hpp"
#include <ripple/utility/bitops.hpp>
#include <algorithm>
#include <vector>

#include <iostream>

namespace ripple {

/// Defines functions for CPUID.
enum CpuidFunction : uint32_t {
  MaxFunction      = 0x00000000,  //!< Max supported function.
  MaxExtFunction   = 0x80000000,  //!< Max supported exented function.
  CoreAndCacheInfo = 0x00000004   //!< Cache and core information.
};

/// Defines the max number of subleaves for cache info.
static constexpr uint32_t max_cache_subleaves = 16;

/// Provides funtionality for cpu related information.
struct CpuInfo {
 private:
  /// Struct to store registers for CPUID.
  struct Regs {
    /// Defines the type of a register.
    using reg_t = uint32_t;
    reg_t eax = 0;    //!< EAX register.
    reg_t ebx = 0;    //!< EBX register.
    reg_t ecx = 0;    //!< ECX register.
    reg_t edx = 0;    //!< EDX register.
  };

  Regs     regs;             //!< Registers to fill when calling cpuid.
  uint32_t max_func     = 0; //!< Maximum supported function.
  uint32_t max_ext_func = 0; //!< Maximum supported extended function.

 public:
  /// Constructor, initialises that max supported values for cpuid.
  CpuInfo() {
    // Determine the max supported function:
    regs.eax = CpuidFunction::MaxFunction;;
    cpuid();
    max_func = regs.eax;

    // Determine the max suported extended function:
    regs.eax = CpuidFunction::MaxExtFunction;
    regs.ecx = 0;
    cpuid();
    max_ext_func = regs.eax;

    printf("Max : %3u Max Ext : %08x\n", 
      max_func, max_ext_func
    );
  }

  /// Creates cache information for the cpu, returning a vector of caches,
  /// ordered by level (lowest level - L1) first.
  auto create_cache_info() {
    std::vector<Cache> caches;

    // CPU does not support cache information, none can be returned.
    if (max_func < CpuidFunction::CoreAndCacheInfo) {
      return caches;
    }

    uint32_t subleaf = 0;
    while (subleaf < max_cache_subleaves) {
      caches.emplace_back();
      auto& cache = caches.back();
      regs.eax    = CpuidFunction::CoreAndCacheInfo;
      regs.ecx    = subleaf;
      cpuid();

      cache.type  = static_cast<Cache::Type>(bits(regs.eax, 0, 4));
      // CPUID reference says that once cpuid returns (0 - Null), then the end
      // of the cache information has been reached.
      if (cache.type == Cache::Type::Null) {
        caches.pop_back();
        break;
      }

      cache.level          = bits(regs.eax, 5 , 7 );
      cache.linesize       = bits(regs.ebx, 0 , 11) + 1;
      cache.partitions     = bits(regs.ebx, 12, 21) + 1;
      cache.assosciativity = bits(regs.ebx, 22, 31) + 1;
      cache.shared_by      = bits(regs.eax, 14, 25) + 1;
      cache.sets           = bits(regs.ecx, 0 , 31) + 1;

      /// Determine the mask for the cache, which can be used to 
      /// determine which processors share the cache.
      uint32_t temp = cache.shared_by - 1, mask_width = 0;
      for (; temp; mask_width++) {
        temp >>= 1;
      }
      cache.mask = (-1) << mask_width;
      subleaf++;
    }

    // Sort caches from lowest level to highest level,
    // and by type for same level.
    std::sort(caches.begin(), caches.end(), 
      [] (const Cache& a, const Cache& b) -> bool {
        return a.level == b.level ? a.type <= b.type : a.level < b.level;
      }
    );
    return caches;
  }

 private:
  /// Performs the cpuid instruction, with the values in eax and ecx as the
  /// input arguments, filling all the registers with the results.
  inline auto cpuid() -> void {
    asm("cpuid"
      : "=a" (regs.eax),
        "=b" (regs.ebx),
        "=c" (regs.ecx),
        "=d" (regs.edx)
      : "0" (regs.eax),
        "2" (regs.ecx)
    );
  }
};

} // namespace ripple

#endif // RIPPLE_ARCH_CPUINFO_HPP


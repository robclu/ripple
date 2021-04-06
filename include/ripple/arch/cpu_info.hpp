/**=--- ripple/arch/cpu_info.hpp --------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  cpu_info.hpp
 * \brief This file defines functionality for cpu information.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ARCH_CPU_INFO_HPP
#define RIPPLE_ARCH_CPU_INFO_HPP

#include "cache.hpp"
#include "cpu_utils.hpp"
#include "proc_info.hpp"
#include <ripple/utility/bitops.hpp>
#include <algorithm>
#include <vector>

namespace ripple {

/** Defines the max number of subleaves for cache info. */
static constexpr uint32_t max_cache_subleaves = 16;

/**
 * Provides funtionality for cpu related information, such as the properties of
 * the caches, number of available cores, packages, cores per package, and the
 * indices of the package, core and thread which hardware threads should use
 * for improved performance.
 */
struct CpuInfo {
 private:
  /*==--- [aliases] --------------------------------------------------------==*/

  // clang-format off
  /** Defines the type of container for caches. */
  using CacheContainer    = std::vector<Cache>;
  /** Defines the type of container for processor information. */
  using ProcInfoContainer = std::vector<ProcInfo>;
  // clang-format on

 public:
  /*==--- [construction] ---------------------------------------------------==*/

  /* Constructor, initialises that max supported values for cpuid. */
  CpuInfo() noexcept {
    // Determine the max supported function:
    regs_.eax = CpuidFunction::MaxFunction;
    cpuid();
    max_func_ = regs_.eax;

    // Determine the max suported extended function:
    regs_.eax = CpuidFunction::MaxExtFunction;
    regs_.ecx = 0;
    cpuid();
    max_ext_func_ = regs_.eax;

    set_max_cores();
    create_cache_info();
    create_proc_info();
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Returns the number of available physical cores.
   */
  auto available_cores() const noexcept -> uint32_t {
    return num_packages_ * cores_per_package_;
  }

  /**
   * Returns the number of packages.
   */
  auto packages() const noexcept -> uint32_t {
    return num_packages_;
  }

  /**
   * Returns the number of cores per package.
   */
  auto cores_per_package() const noexcept -> uint32_t {
    return cores_per_package_;
  }

  /**
   * Returns a reference to the container of caches.
   */
  auto cache_info() noexcept -> CacheContainer& {
    return caches_;
  }

  /**
   * Returns a reference to the container of processor information.
   */
  auto processor_info() noexcept -> ProcInfoContainer& {
    return proc_info_;
  }

 private:
  /** Defines topology levels. */
  enum TopoLevel {
    Invalid = 0x0, //!< Invalid level.
    Thread  = 0x1, //!< Thread level topology,
    Core    = 0x2  //!< Core level topology.
  };

  /** Defines functions for CPUID. */
  enum CpuidFunction : uint32_t {
    MaxFunction       = 0x00000000, //!< Max supported function.
    MaxExtFunction    = 0x80000000, //!< Max supported exented function.
    CoreAndCacheInfo  = 0x00000004, //!< Cache and core information.
    ProcessorTopology = 0x0000000B  //!< Topology information
  };

  /** Struct to store registers for CPUID. */
  struct Regs {
    /** Defines the type of a register. */
    using Reg = uint32_t;

    Reg eax = 0; //!< EAX register.
    Reg ebx = 0; //!< EBX register.
    Reg ecx = 0; //!< ECX register.
    Reg edx = 0; //!< EDX register.
  };

  /** Defines the max number of subleaves for cache info. */
  static constexpr uint32_t max_cache_subleaves = 16;

  Regs     regs_;                  //!< Registers to fill when calling cpuid.
  uint32_t max_func_          = 0; //!< Maximum supported function.
  uint32_t max_ext_func_      = 0; //!< Maximum supported extended function.
  uint32_t max_cores_         = 0; //!< Max number of cores in package.
  uint32_t num_packages_      = 0; //!< Number of packages on the node.
  uint32_t cores_per_package_ = 0; //!< Number of cores in each package.
  uint32_t threads_per_core_  = 0; //!< Number of logical threads per core.

  CacheContainer    caches_;    //!< Container of cpu caches.
  ProcInfoContainer proc_info_; //!< Container of processor information.

  /*==--- [methods] --------------------------------------------------------==*/

  /**
   * Creates cache information for the cpu, filling the cache container, and
   * ordering it from L1 cache upwards.
   */
  auto create_cache_info() noexcept -> void {
    // CPU does not support cache information, none can be returned.
    if (max_func_ < CpuidFunction::CoreAndCacheInfo) {
      return;
    }

    uint32_t subleaf = 0;
    while (subleaf < max_cache_subleaves) {
      auto& cache = caches_.emplace_back();
      regs_.eax   = CpuidFunction::CoreAndCacheInfo;
      regs_.ecx   = subleaf;
      cpuid();

      cache.type = static_cast<Cache::Type>(bits(regs_.eax, 0, 4));
      // CPUID reference says that once cpuid returns (0 - Null), then the end
      // of the cache information has been reached.
      if (cache.type == Cache::Type::Null) {
        caches_.pop_back();
        break;
      }

      cache.level          = bits(regs_.eax, 5, 7);
      cache.linesize       = bits(regs_.ebx, 0, 11) + 1;
      cache.partitions     = bits(regs_.ebx, 12, 21) + 1;
      cache.assosciativity = bits(regs_.ebx, 22, 31) + 1;
      cache.shared_by      = bits(regs_.eax, 14, 25) + 1;
      cache.sets           = bits(regs_.ecx, 0, 31) + 1;

      // Determine the mask for the cache, which can be used to
      // determine which processors share the cache.
      uint32_t temp = cache.shared_by - 1, mask_width = 0;
      for (; temp; mask_width++) {
        temp >>= 1;
      }
      cache.mask = (-1) << mask_width;
      subleaf++;
    }

    // Sort caches from lowest level to highest level,
    // and by type for same level.
    std::sort(
      caches_.begin(),
      caches_.end(),
      [](const Cache& a, const Cache& b) -> bool {
        return a.level == b.level ? a.type <= b.type : a.level < b.level;
      });
  }

  /**
   * Creates proccessor information, filling each of the processor info structs
   * with the package, core, and thread index in the topology.
   */
  auto create_proc_info() noexcept -> void {
    for (uint32_t i = 0; i < max_cores_; ++i) {
      create_proc_info_for_proc(i);
    }

    // Find max index for package, core and thred from all procsessors.
    for (auto& info : proc_info_) {
      if (info.is_invalid()) {
        continue;
      }
      if (info.package > num_packages_) {
        num_packages_ = info.package;
      }
      if (info.core > cores_per_package_) {
        cores_per_package_ = info.core;
      }
      if (info.thread > threads_per_core_) {
        threads_per_core_ = info.thread;
      }
    }
    num_packages_ += 1;
    cores_per_package_ += 1;
    threads_per_core_ += 1;
  }

  /**
   * Create processor information for the processor with index proc_index.
   * \param proc_index The index of the processor to creat the info for.
   */
  auto create_proc_info_for_proc(uint32_t proc_index) noexcept -> void {
    if (proc_index > max_cores_) {
      return;
    }
    set_affinity(proc_index);
    auto& info = proc_info_.emplace_back();

    uint32_t apic = 0, apic_width = 0, level_type = 0, mask = 0,
             select_mask = 0, topo_level = 0, prev_mask_width = 0;

    bool next_level_invalid = false;
    while (!next_level_invalid) {
      regs_.eax = CpuidFunction::ProcessorTopology;
      regs_.ecx = topo_level++;
      cpuid();
      next_level_invalid = regs_.eax == 0 && regs_.ebx == 0;

      apic_width = bits(regs_.eax, 0, 4);
      level_type = bits(regs_.ecx, 8, 15);
      apic       = bits(regs_.edx, 0, 31);
      mask       = bitmask(apic_width);
      if (level_type == TopoLevel::Thread) {
        select_mask     = mask;
        info.thread     = apic & select_mask;
        prev_mask_width = apic_width;
        continue;
      }
      if (level_type == TopoLevel::Core) {
        select_mask = mask ^ select_mask;
        info.core   = (apic & select_mask) >> prev_mask_width;

        // For the package:
        select_mask  = (-1) << apic_width;
        info.package = (apic & select_mask) >> apic_width;
      }
    }
  }

  /**
   * Sets the max number of cores for the CPU.
   */
  auto set_max_cores() noexcept -> void {
    if (max_func_ >= CpuidFunction::CoreAndCacheInfo) {
      regs_.eax = CpuidFunction::CoreAndCacheInfo;
      regs_.ecx = 0;
      cpuid();
      max_cores_ = bits(regs_.eax, 26, 31) + 1;
    }
  }

  /**
   * Performs the cpuid instruction, with the values in eax and ecx as the
   * input arguments, filling all the registers with the results.
   */
  inline auto cpuid() noexcept -> void {
    asm("cpuid"
        : "=a"(regs_.eax), "=b"(regs_.ebx), "=c"(regs_.ecx), "=d"(regs_.edx)
        : "0"(regs_.eax), "2"(regs_.ecx));
  }
};

} // namespace ripple

#endif // RIPPLE_ARCH_CPU_INFO_HPP

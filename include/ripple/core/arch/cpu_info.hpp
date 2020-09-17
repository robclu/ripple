//==--- ripple/core/arch/cpu_info.hpp --------------------------- -*- C++ -*-
//---==//
//
//                                Ripple
//
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cpu_info.hpp
/// \brief This file defines functionality for cpu information.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ARCH_CPU_INFO_HPP
#define RIPPLE_ARCH_CPU_INFO_HPP

#include "cache.hpp"
#include "cpu_utils.hpp"
#include "proc_info.hpp"
#include <ripple/core/utility/bitops.hpp>
#include <algorithm>
#include <vector>

namespace ripple {

/// Defines the max number of subleaves for cache info.
static constexpr uint32_t max_cache_subleaves = 16;

/// Provides funtionality for cpu related information, such as the properties of
/// the caches, number of available cores, packages, cores per package, and the
/// indices of the package, core and thread which hardware threads should use
/// for improved performance.
struct CpuInfo {
 private:
  //==--- [aliases] --------------------------------------------------------==//
  /// Defines the type of container for caches.
  using cache_container_t = std::vector<Cache>;
  /// Defines the type of container for processor information.
  using proc_info_container_t = std::vector<ProcInfo>;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Constructor, initialises that max supported values for cpuid.
  CpuInfo() {
    // Determine the max supported function:
    _regs.eax = CpuidFunction::MaxFunction;
    ;
    cpuid();
    _max_func = _regs.eax;

    // Determine the max suported extended function:
    _regs.eax = CpuidFunction::MaxExtFunction;
    _regs.ecx = 0;
    cpuid();
    _max_ext_func = _regs.eax;

    set_max_cores();
    create_cache_info();
    create_proc_info();
  }

  //==--- [interface] ------------------------------------------------------==//

  /// Returns the number of available physical cores.
  auto available_cores() const -> uint32_t {
    return _num_packages * _cores_per_package;
  }

  /// Returns the number of packages.
  auto packages() -> uint32_t {
    return _num_packages;
  }

  /// Returns the number of cores per package.
  auto cores_per_package() -> uint32_t {
    return _cores_per_package;
  }

  /// Returns a reference to the container of caches.
  auto cache_info() -> cache_container_t& {
    return _caches;
  }

  /// Returns a reference to the container of processor information.
  auto processor_info() -> proc_info_container_t& {
    return _proc_info;
  }

 private:
  //==--- [enums] ----------------------------------------------------------==//

  /// Defines topology levels.
  enum TopoLevel {
    Invalid = 0x0, //!< Invalid level.
    Thread  = 0x1, //!< Thread level topology,
    Core    = 0x2  //!< Core level topology.
  };

  /// Defines functions for CPUID.
  enum CpuidFunction : uint32_t {
    MaxFunction       = 0x00000000, //!< Max supported function.
    MaxExtFunction    = 0x80000000, //!< Max supported exented function.
    CoreAndCacheInfo  = 0x00000004, //!< Cache and core information.
    ProcessorTopology = 0x0000000B  //!< Topology information
  };

  //==--- [structs] --------------------------------------------------------==//

  /// Struct to store registers for CPUID.
  struct Regs {
    /// Defines the type of a register.
    using reg_t = uint32_t;
    reg_t eax   = 0; //!< EAX register.
    reg_t ebx   = 0; //!< EBX register.
    reg_t ecx   = 0; //!< ECX register.
    reg_t edx   = 0; //!< EDX register.
  };

  //==--- [constants] ------------------------------------------------------==//

  /// Defines the max number of subleaves for cache info.
  static constexpr uint32_t max_cache_subleaves = 16;

  //==--- [members] --------------------------------------------------------==//

  Regs     _regs;                  //!< Registers to fill when calling cpuid.
  uint32_t _max_func          = 0; //!< Maximum supported function.
  uint32_t _max_ext_func      = 0; //!< Maximum supported extended function.
  uint32_t _max_cores         = 0; //!< Max number of cores in package.
  uint32_t _num_packages      = 0; //!< Number of packages on the node.
  uint32_t _cores_per_package = 0; //!< Number of cores in each package.
  uint32_t _threads_per_core  = 0; //!< Number of logical threads per core.

  cache_container_t     _caches;    //!< Container of cpu caches.
  proc_info_container_t _proc_info; //!< Container of processor information.

  //==--- [methods] --------------------------------------------------------==//

  /// Creates cache information for the cpu, filling the cache container, and
  /// ordering it from L1 cache upwards.
  auto create_cache_info() -> void {
    // CPU does not support cache information, none can be returned.
    if (_max_func < CpuidFunction::CoreAndCacheInfo) {
      return;
    }

    uint32_t subleaf = 0;
    while (subleaf < max_cache_subleaves) {
      auto& cache = _caches.emplace_back();
      _regs.eax   = CpuidFunction::CoreAndCacheInfo;
      _regs.ecx   = subleaf;
      cpuid();

      cache.type = static_cast<Cache::Type>(bits(_regs.eax, 0, 4));
      // CPUID reference says that once cpuid returns (0 - Null), then the end
      // of the cache information has been reached.
      if (cache.type == Cache::Type::Null) {
        _caches.pop_back();
        break;
      }

      cache.level          = bits(_regs.eax, 5, 7);
      cache.linesize       = bits(_regs.ebx, 0, 11) + 1;
      cache.partitions     = bits(_regs.ebx, 12, 21) + 1;
      cache.assosciativity = bits(_regs.ebx, 22, 31) + 1;
      cache.shared_by      = bits(_regs.eax, 14, 25) + 1;
      cache.sets           = bits(_regs.ecx, 0, 31) + 1;

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
    std::sort(
      _caches.begin(),
      _caches.end(),
      [](const Cache& a, const Cache& b) -> bool {
        return a.level == b.level ? a.type <= b.type : a.level < b.level;
      });
  }

  /// Creates proccessor information, filling each of the processor info structs
  /// with the package, core, and thread index in the topology.
  auto create_proc_info() -> void {
    for (uint32_t i = 0; i < _max_cores; ++i) {
      create_proc_info_for_proc(i);
    }

    // Find max index for package, core and thred from all procsessors.
    for (auto& info : _proc_info) {
      if (info.is_invalid()) {
        continue;
      }
      if (info.package > _num_packages) {
        _num_packages = info.package;
      }
      if (info.core > _cores_per_package) {
        _cores_per_package = info.core;
      }
      if (info.thread > _threads_per_core) {
        _threads_per_core = info.thread;
      }
    }
    _num_packages += 1;
    _cores_per_package += 1;
    _threads_per_core += 1;
  }

  /// Create processor information for the processor with index \p proc_index.
  /// \p proc_index The index of the processor to creat the info for.
  auto create_proc_info_for_proc(uint32_t proc_index) -> void {
    if (proc_index > _max_cores) {
      return;
    }
    set_affinity(proc_index);
    auto& info = _proc_info.emplace_back();

    uint32_t apic = 0, apic_width = 0, level_type = 0, mask = 0,
             select_mask = 0, topo_level = 0, prev_mask_width = 0;

    bool next_level_invalid = false;
    while (!next_level_invalid) {
      _regs.eax = CpuidFunction::ProcessorTopology;
      _regs.ecx = topo_level++;
      cpuid();
      next_level_invalid = _regs.eax == 0 && _regs.ebx == 0;

      apic_width = bits(_regs.eax, 0, 4);
      level_type = bits(_regs.ecx, 8, 15);
      apic       = bits(_regs.edx, 0, 31);
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

  /// Sets the max number of cores for the CPU.
  auto set_max_cores() -> void {
    if (_max_func >= CpuidFunction::CoreAndCacheInfo) {
      _regs.eax = CpuidFunction::CoreAndCacheInfo;
      _regs.ecx = 0;
      cpuid();
      _max_cores = bits(_regs.eax, 26, 31) + 1;
    }
  }

  /// Performs the cpuid instruction, with the values in eax and ecx as the
  /// input arguments, filling all the registers with the results.
  inline auto cpuid() -> void {
    asm("cpuid"
        : "=a"(_regs.eax), "=b"(_regs.ebx), "=c"(_regs.ecx), "=d"(_regs.edx)
        : "0"(_regs.eax), "2"(_regs.ecx));
  }
};

} // namespace ripple

#endif // RIPPLE_ARCH_CPU_INFO_HPP

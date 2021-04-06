/**=--- ripple/allocation/multiarch_allocator.hpp ---------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  multiarch_allocator.hpp
 * \brief This file defines an allocator for cpus and gpus.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ALLOCATION_MULTIARCH_ALLOCATOR_HPP
#define RIPPLE_ALLOCATION_MULTIARCH_ALLOCATOR_HPP

#include "arena.hpp"
#include "linear_allocator.hpp"
#include <ripple/arch/topology.hpp>
#include <ripple/utility/spinlock.hpp>
#include <mutex>
#include <vector>

namespace ripple {

/**
 * Defines the architextures which can be allocated for.
 */
enum class AllocArch {
  cpu = 0, //!< Allocate for the host.
  gpu = 1, //!< Allcoate for the device.
};

/**
 * Defines a simple allocator for either the cpu or the gpu.
 * \tparam Arch The architecture to allocate for.
 * \tparam LockingPolicy The locking policy for the allocator.
 */
template <AllocArch Arch, typename LockingPolicy = Spinlock>
class alignas(avoid_false_sharing_size) SimpleAllocator {
  /** True if the architecture is the cpu. */
  static constexpr bool cpu_allocator = Arch == AllocArch::cpu;

  /**
   * Defines the type of the arena for allocation.
   */
  using ArchArena = std::conditional_t<cpu_allocator, HeapArena, GpuHeapArena>;

  ArchArena       arena_;     //!< Arena for the device.
  LinearAllocator allocator_; //!< Allocator for the device.
  LockingPolicy   lock_;      //!< Lock for the allocator.

  /**
   * Defines the type of the guard for locking when allocating. We use a
   * spinlock here because benchmarks have shown how fast it is, even compared
   * to atomics.
   */
  using Guard = std::lock_guard<LockingPolicy>;

 public:
  /**
   * Constructor to set the device id for the allocator.
   * \param dev_id The id of the device to allocate for.
   */
  template <
    AllocArch ArchType                                = Arch,
    std::enable_if_t<ArchType == AllocArch::cpu, int> = 0>
  SimpleAllocator() noexcept
  : arena_{}, allocator_(arena_.begin(), arena_.end()) {}

  /**
   * Constructor to set the device id for the allocator.
   * \param dev_id The id of the device to allocate for.
   */
  template <
    AllocArch ArchType                                = Arch,
    std::enable_if_t<ArchType == AllocArch::gpu, int> = 0>
  SimpleAllocator(size_t dev_id) noexcept
  : arena_{dev_id}, allocator_(arena_.begin(), arena_.end()) {}

  /**
   * Reserves The given number of bytes for the allocator.
   * \param bytes The number of bytes to reserve.
   */
  auto reserve(size_t bytes) -> void {
    Guard g(lock_);
    arena_.resize(bytes);
    allocator_.reset(arena_.begin(), arena_.end());
  }

  /**
   * Allocates the given amount of memory.
   * \param size The number of bytes to allocate.
   * \return A pointer to the new memory.
   */
  auto alloc(size_t size, size_t alignment) -> void* {
    if (allocator_.capacity() < size) {
      // log_error("Not enough memory remaining for device allocation");
      printf(
        "Allocator too small, cap : %4lu, req : %4lu\n",
        allocator_.capacity(),
        size);
      return nullptr;
    }
    Guard g(lock_);
    return allocator_.alloc(size, alignment);
  }

  /**
   * Frees the memory pointed to by the pointer. This does nothing as of right
   * now, this allocator can only reset.
   * \param ptr       The pointer to free.
   */
  auto free(void* ptr) -> void {}

  /**
   * Resets the allocator.
   */
  auto reset() noexcept -> void {
    Guard g(lock_);
    allocator_.reset();
  }
};

/**
 * MultiarchAllocator which allows for allocations on the host and multiple
 * devices. Each allocation is done per thread or per device, and is
 * thread-safe.
 *
 * Each allocator is simply a linear allocator, which should be reset once the
 * memory must be freed.
 */
class MultiarchAllocator {
 public:
  /**
   * Constructor to set the number of allocators and the area for each of the
   * allocators.
   * \param num_gpus The number of gpu allocators to create.
   */
  MultiarchAllocator(size_t num_gpus) {
    for (size_t i = 0; i < num_gpus; ++i) {
      gpu_allocators_.emplace_back(i);
    }
  }

  /**
   * Destructor, frees all the allocated memory for the device.
   */
  ~MultiarchAllocator() noexcept {
    for (auto& alloc : gpu_allocators_) {
      alloc.reset();
    }
    cpu_allocator_.reset();
  }

  /** Defines the type of the gpu allocator. */
  using GpuAllocator = SimpleAllocator<AllocArch::gpu>;
  /** Defines the type of the cpu allocator. */
  using CpuAllocator = SimpleAllocator<AllocArch::cpu>;
  /** Defines the type of the container for per gpu allocators. */
  using GpuAllocators = std::vector<GpuAllocator>;

  /**
   * Gets a reference to the allocator with the given gpu id.
   * \param device_id The id of the device to get the allocator for.
   * \return A reference to the allocator for the gpu.
   */
  auto gpu_allocator(size_t gpu_id) noexcept -> GpuAllocator& {
    return gpu_allocators_[gpu_id];
  }

  /**
   * Gets a reference to the allocator with the given thread id.
   * \param thread_id The id of the thread to get the allocator for.
   * \return A reference to the allocator for the gpu.
   */
  auto cpu_allocator() noexcept -> CpuAllocator& {
    return cpu_allocator_;
  }

  /**
   * Reserves the given number of bytes for each of the gpu allocators.
   * \param bytes_per_gpu The amount of memory to reserve for each gpu
   *        allocator.
   */
  auto reserve_gpu(size_t bytes_per_gpu) -> void {
    for (auto& gpu_alloc : gpu_allocators_) {
      gpu_alloc.reserve(bytes_per_gpu);
    }
  }

  /**
   * Reserves the given number of bytes for the cpu allocator.
   * \param bytes The amount of memory to reserve for the cpu allocator.
   */
  auto reserve_cpu(size_t bytes) -> void {
    cpu_allocator_.reserve(bytes);
  }

  /**
   * Resets all the allocators for each gpu.
   */
  auto reset_gpu() noexcept -> void {
    for (auto& alloc : gpu_allocators_) {
      alloc.reset();
    }
  }

  /**
   * Resets all the allocators for each cpu.
   */
  auto reset_cpu() noexcept -> void {
    cpu_allocator_.reset();
  }

  /**
   * Resets all gpu and cpu allocators.
   */
  auto reset() noexcept -> void {
    reset_gpu();
    reset_cpu();
  }

 private:
  GpuAllocators gpu_allocators_; //!< Allocators for the gpus.
  CpuAllocator  cpu_allocator_;  //!< Allocator for the cpu.
};

/**
 * Gets a reference to the multiarch allocator.
 * \return A reference to the multicarh allocator.
 */
static inline auto multiarch_allocator() noexcept -> MultiarchAllocator& {
  static MultiarchAllocator allocator(topology().num_gpus());
  return allocator;
}

} // namespace ripple

#endif // RIPPLE_ALLOCATION_MULTIARCH_ALLOCATOR_HPP
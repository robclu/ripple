/**=--- ripple/allocation/allocator.hpp -------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  allocator.hpp
 * \brief This file defines a composable allocator.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ALLOCATION_ALLOCATOR_HPP
#define RIPPLE_ALLOCATION_ALLOCATOR_HPP

#include "aligned_heap_allocator.hpp"
#include "arena.hpp"
#include "pool_allocator.hpp"
#include <mutex>

namespace ripple {

/**
 * Default locking implementation which does no locking.
 */
struct VoidLock {
  /** Does nothing when called. */
  auto lock() noexcept -> void {}

  /** Does nothing when called. */
  auto unlock() noexcept -> void {}
};

/*==--- [forward declarations & aliases] -----------------------------------==*/

/**
 * The Allocator type is a simple implementation which allows an allocator to
 * be composed of other allocators, to create allocators with useful properties
 * for different contexts.
 *
 * The allocator will always try to allocate from the primary allocator, unless
 * the primary allocation fails, in which case it will allocate from the
 * fallback allocator.
 *
 * All allocation and free operations are locked, using the locking
 * policy provided. The default locking policy is to not lock.
 *
 * \tparam PrimaryAllocator  The type of the primary allocator.
 * \tparam Arena             The type of the arena for the allocator.
 * \tparam FallbackAllocator The type of the fallback allocator.
 * \tparam LockingPolicy     The type of the locking policy.
 */
template <
  typename PrimaryAllocator,
  typename Arena             = HeapArena,
  typename FallbackAllocator = AlignedHeapAllocator,
  typename LockingPolicy     = VoidLock>
class Allocator;

/**
 * Defines an object pool allocator for objects of type T, which is by default
 * not thread safe.
 * \tparam T The type of the objects to allocate from the pool.
 */
template <
  typename T,
  typename LockingPolicy = VoidLock,
  typename Arena         = HeapArena>
using ObjectPoolAllocator = Allocator<
  PoolAllocator<sizeof(T), std::max(alignof(T), alignof(Freelist)), Freelist>,
  Arena,
  AlignedHeapAllocator,
  LockingPolicy>;

/**
 * Defines an object pool allocator for objects of type T, which is
 * thread-safe.
 * \tparam T The type of the objects to allocate from the pool.
 */
template <typename T, typename Arena = HeapArena>
using ThreadSafeObjectPoolAllocator = Allocator<
  PoolAllocator<
    sizeof(T),
    std::max(alignof(T), alignof(ThreadSafeFreelist)),
    ThreadSafeFreelist>,
  Arena,
  AlignedHeapAllocator,
  VoidLock>;

/*==--- [implementation] ---------------------------------------------------==*/

/**
 * The Allocator type is a simple implementation which allows an allocator to
 * be composed of other allocators, to create allocators with useful
 *  properties for different contexts.
 *
 * The allocator will always try to allocate from the primary allocator,
 * unless the primary allocation fails, in which case it will allocate from
 * the fallback allocator.
 *
 * \note The fallback allocator should always success unless there is no
 *       system memory left to allocate from.
 *
 * All allocation and free operations are locked, using the locking
 * policy provided. The default locking policy is to not lock.
 *
 * \tparam PrimaryAllocator  The type of the primary allocator.
 * \tparam Arena             The type of the arena for the allocator.
 * \tparam FallbackAllocator The type of the fallback allocator.
 * \tparam LockingPolicy     The type of the locking policy.
 */
template <
  typename PrimaryAllocator,
  typename Arena,
  typename FallbackAllocator,
  typename LockingPolicy>
class Allocator {
  static_assert(
    std::is_trivially_constructible_v<FallbackAllocator>,
    "Fallback allocator must be trivially constructible!");

 public:
  /** Returns true if the arena has a contexpr size. */
  static constexpr bool contexpr_arena_size = Arena::constexpr_size;

  /** Defines the type of the lock guard. */
  using Guard = std::lock_guard<LockingPolicy>;

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Constructor which sets the size of the arena, if the arena requires a
   * size, and forwards the args to the primary allocator for construction.
   *
   * \note If the arena has a constant size, then it will be created with that
   *       size and the size argument will be ignored.
   *
   * \param  size  The size of the arena.
   * \param  args  The arguments fro the primary allocator.
   * \tparam Args  The types of arguments for the primary allocator.
   */
  template <typename... Args>
  Allocator(size_t size, Args&&... args)
  : arena_(size), primary_(arena_, ripple_forward(args)...) {}

  /**
   * Default destructor -- composed allocators know how to clean themselves up.
   */
  ~Allocator() noexcept = default;

  // clang-format off
  /**
   * Move constructor, defaulted.
   * \param other The other allocator to move into this one.
   */
  Allocator(Allocator&& other) noexcept                    = default;
  /** 
   * Move assignment, defaulted.
   * \param other The other allocator to move into this one.
   */
  auto operator=(Allocator&& other) noexcept -> Allocator& = default;

  /*==--- [deleted] --------------------------------------------------------==*/

  /** Copy constructor -- deleted, allocator can't be copied. */
  Allocator(const Allocator&)      = delete;
  /** Copy assignment -- deleted, allocator can't be copied. */
  auto operator=(const Allocator&) = delete;
  // clang-format on

  /*==--- [alloc/free interface] -------------------------------------------==*/

  /**
   * Allocates the given number of bytes of memory with given alignment.
   * \param size      The size of the memory to allocate.
   * \param alignment The alignment of the allocation.
   * \return A pointer to the allocated region, or nullptr.
   */
  auto alloc(size_t size, size_t alignment = alignof(std::max_align_t)) noexcept
    -> void* {
    Guard g(lock_);
    void* ptr = primary_.alloc(size, alignment);
    if (!ptr) {
      ptr = fallback_.alloc(size, alignment);
    }
    return ptr;
  }

  /**
   * Frees the memory pointed to by ptr.
   * \param ptr The pointer to the memory to free.
   */
  auto free(void* ptr) noexcept -> void {
    if (!ptr) {
      return;
    }

    Guard g(lock_);
    if (primary_.owns(ptr)) {
      primary_.free(ptr);
      return;
    }

    fallback_.free(ptr);
  }

  /**
   * Frees the memory pointed to by the pointer.
   * \param ptr  The pointer to the memory to free.
   * \param size The size of the memory to free.
   */
  auto free(void* ptr, size_t size) noexcept -> void {
    if (!ptr) {
      return;
    }

    Guard g(lock_);
    if (primary_.owns(ptr)) {
      primary_.free(ptr, size);
      return;
    }
    fallback_.free(ptr, size);
  }

  /**
   * Resets the primary and fallback allocators.
   */
  auto reset() noexcept -> void {
    Guard g(lock_);
    primary_.reset();
  }

  /*==--- [create/destroy interface] ---------------------------------------==*/

  /**
   * Allocates and constructs an object of type T. If this is used, then
   * destroy should be used to destruct and free the object, rather than free.
   * \param args The arguments for constructing the objects.
   * \tparam T    The type of the object to allocate.
   * \tparam Args The types of the arguments for constructing T.
   * \return A pointer to the created object, or a nullptr.
   */
  template <typename T, typename... Args>
  auto create(Args&&... args) noexcept -> T* {
    constexpr size_t size      = sizeof(T);
    constexpr size_t alignment = alignof(T);
    void* const      ptr       = alloc(size, alignment);

    return ptr ? new (ptr) T(ripple_forward(args)...) : nullptr;
  }

  /**
   * Recycles the pointed to object, destructing it and then releasing the
   * memory back to the allocator.
   * \param  ptr A pointer to the object to destroy.
   * \tparam T   The type of the object.
   */
  template <typename T>
  auto recycle(T* ptr) noexcept -> void {
    if (!ptr) {
      return;
    }

    constexpr size_t size = sizeof(T);
    ptr->~T();
    free(static_cast<void*>(ptr), size);
  }

 private:
  Arena             arena_;    //!< The type of the arena.
  PrimaryAllocator  primary_;  //!< The primary allocator.
  FallbackAllocator fallback_; //!< The fallback allocator.
  LockingPolicy     lock_;     //!< The locking implementation.
};

} // namespace ripple

#endif // RIPPLE_ALLOCATION_ALLOCATOR_HP
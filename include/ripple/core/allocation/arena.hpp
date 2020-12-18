//==--- ripple/core/memory/arena.hpp ----------------------- -*- C++ -*- ---==//
//
//                            Ripple - Core
//
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  arena.hpp
/// \brief This file defines memory arena implementations.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALLOCATION_ARENA_HPP
#define RIPPLE_ALLOCATION_ARENA_HPP

#include "../arch/gpu_utils.hpp"
#include "../utility/memory.hpp"
#include <type_traits>

namespace ripple {

/**
 * Defines a stack-based memory arena of a specific size.
 * \tparam Size The size of the stack for the arena.
 */
template <size_t Size>
class StackArena {
  /** Defines the size of the stack for the arena. */
  static constexpr size_t stack_size = Size;

 public:
  // clang-format off
  /** Returns that the allocator has a constexpr size. */
  static constexpr bool contexpr_size   = true;
  /** Returns that the arena is valid on the device. */
  static constexpr bool valid_on_device = true;
  /** Returns that the arena is valid on the host. */
  static constexpr bool valid_on_host   = true;
  // clang-format on

  using Ptr      = void*;       //!< Pointer type.
  using ConstPtr = const void*; //!< Const pointer type.

  /**
   * Constructor which takes the size of the arena. This is provided so that
   * arenas have the same interface.
   * \param size The size of the arena.
   *
   */
  ripple_host_device StackArena(size_t size = 0) noexcept {}

  /**
   * Gets a pointer to the beginning of the arena.
   * \return A pointer to the beginning of the arena.
   */
  ripple_nodiscard ripple_host_device auto begin() const noexcept -> ConstPtr {
    return static_cast<ConstPtr>(&buffer_[0]);
  }

  /**
   * Gets a pointer to the end of the arena.
   * \return A pointer to the end of the arena.
   */
  ripple_nodiscard ripple_host_device auto end() const noexcept -> ConstPtr {
    return static_cast<ConstPtr>(&buffer_[stack_size]);
  }

  /**
   * Gets the size of the arena.
   * \return The size of the arena.
   */
  ripple_nodiscard ripple_host_device constexpr auto
  size() const noexcept -> size_t {
    return stack_size;
  }

 private:
  char buffer_[stack_size]; //!< The buffer for the stack.
};

/**
 * Defines a heap-based arena for the host.
 */
struct HeapArena {
 public:
  // clang-format off
  /** Returns that the allocator does not have a constexpr size. */
  static constexpr bool constexpr_size  = false;
  /** Returns that the arena is not valid on the device. */
  static constexpr bool valid_on_device = false;
  /** Returns that the arena is valid on the host. */
  static constexpr bool valid_on_host   = true;
  // clang-format on

  using Ptr      = void*; //!< Pointer type.
  using ConstPtr = void*; //!< Const pointer type.

  /**
   * Initializes the arena with a specific size.
   * \param size The size of the arena.
   */
  explicit HeapArena(size_t size = 0) {
    if (size) {
      cpu::allocate_host_pinned(&start_, size);
      end_ = offset_ptr(start_, size);
    }
  }

  /**
   * Destructor to release the memory.
   */
  ~HeapArena() noexcept {
    if (start_ != nullptr) {
      cpu::free_host_pinned(start_);
      start_ = nullptr;
      end_   = nullptr;
    }
  }

  // clang-format off
  /** Copy constructor -- deleted. */
  HeapArena(const HeapArena&)     = delete;
  /** Move constructor -- deleted. */
  HeapArena(HeapArena&&) noexcept = default;

  /** Copy assignment operator -- deleted. */
  auto operator=(const HeapArena&)                   = delete;
  /** Move assignment operator -- deleted. */
  auto operator=(HeapArena&&) noexcept -> HeapArena& = default;
  // clang-format on

  /**
   * Gets a pointer to the beginning of the arena.
   * \return A pointer to the beginning of the arena.
   */
  ripple_nodiscard auto begin() const noexcept -> ConstPtr {
    return start_;
  }

  /**
   * Gets a pointer to the end of the arena.
   * \return A pointer to the end of the arena.
   */
  ripple_nodiscard auto end() const noexcept -> ConstPtr {
    return end_;
  }

  /**
   * Resizes the area.
   *
   * \note This may be expensive since it needs to copy the already allocated
   *       memory in the new space.
   *
   * \param size The new size of the area.
   */
  auto resize(size_t new_size) -> void {
    if (new_size < size()) { return; }
    void* new_ptr = nullptr;
    cpu::allocate_host_pinned(&new_ptr, new_size);
    if (start_) {
      memcpy(new_ptr, start_, size());
      cpu::free_host_pinned(start_);
    }
    start_ = new_ptr;
    end_   = offset_ptr(start_, new_size);
  }

  /**
   * Gets the size of the arena.
   * \return A size of the arena, in bytes.
   */
  ripple_nodiscard auto size() const noexcept -> size_t {
    return uintptr_t(end_) - uintptr_t(start_);
  }

 private:
  void* start_ = nullptr; //!< Pointer to the heap data.
  void* end_   = nullptr; //!< Pointer to the heap data.
};

/**
 * Defines a heap-based arena for the gpu.
 */
struct GpuHeapArena {
 public:
  // clang-format off
  /** Returns that the allocator does not have a constexpr size. */
  static constexpr bool constexpr_size  = false;
  /** Returns that the arena is not valid on the device. */
  static constexpr bool valid_on_device = true;
  /** Returns that the arena is valid on the host. */
  static constexpr bool valid_on_host   = false;
  // clang-format on

  using Ptr      = void*; //!< Pointer type.
  using ConstPtr = void*; //!< Const pointer type.

  /**
   * Initializes the arena with a specific size.
   * \param size The size of the arena.
   */
  explicit GpuHeapArena(size_t id, size_t size = 0) : id_{id} {
    if (size) {
      gpu::set_device(id_);
      gpu::allocate_device(&start_, size);
      end_ = offset_ptr(start_, size);
    }
  }

  /**
   * Destructor to release the memory.
   */
  ~GpuHeapArena() noexcept {
    cleanup();
  }

  // clang-format off
  /** Copy constructor -- deleted. */
  GpuHeapArena(const GpuHeapArena&)     = delete;
  /** Move constructor -- defaulted. */
  GpuHeapArena(GpuHeapArena&&) noexcept = default;

  /** Copy assignment operator -- deleted. */
  auto operator=(const GpuHeapArena&)                      = delete;
  /** Move assignment operator -- defauled. */
  auto operator=(GpuHeapArena&&) noexcept -> GpuHeapArena& = default;
  // clang-format on

  /**
   * Gets a pointer to the beginning of the arena.
   * \return A pointer to the beginning of the arena.
   */
  ripple_nodiscard auto begin() const noexcept -> ConstPtr {
    return start_;
  }

  /**
   * Gets a pointer to the end of the arena.
   * \return A pointer to the end of the arena.
   */
  ripple_nodiscard auto end() const noexcept -> ConstPtr {
    return end_;
  }

  /**
   * Resizes the area.
   *
   * \note This may be expensive since it needs to copy the already allocated
   *       memory in the new space.
   *
   * \param size The new size of the area.
   */
  auto resize(size_t new_size) -> void {
    if (new_size < size()) { return; }
    void* new_ptr = nullptr;
    gpu::set_device(id_);
    gpu::allocate_device(&new_ptr, new_size);
    if (start_) {
      gpu::memcpy_device_to_device_async(new_ptr, start_, size());
      gpu::free_device(start_);
    }
    start_ = new_ptr;
    end_   = offset_ptr(start_, new_size);
  }

  /**
   * Gets the size of the arena.
   * \return A size of the arena, in bytes.
   */
  ripple_nodiscard auto size() const noexcept -> size_t {
    return uintptr_t(end_) - uintptr_t(start_);
  }

 private:
  void*  start_ = nullptr; //!< Pointer to the heap data.
  void*  end_   = nullptr; //!< Pointer to the heap data.
  size_t id_    = 0;       //!< Id of the device for the arena.

  /**
   * Cleans up the allocator resources.
   */
  auto cleanup() noexcept -> void {
    if (start_) {
      gpu::set_device(id_);
      gpu::synchronize_device();
      gpu::free_device(start_);
      start_ = nullptr;
      end_   = nullptr;
    }
  }
};

/*==--- [aliases] ----------------------------------------------------------==*/

/** Defines the default size for a stack arena. */
static constexpr size_t default_stack_arena_size = 1024;

/** Defines the type for a default stack arena. */
using DefaultStackArena = StackArena<default_stack_arena_size>;

/**
 * Defines a valid type if the Arena has a contexpr size.
 * \tparam Arena The arena to base the enable on.
 */
template <typename Arena>
using arena_constexpr_size_enable_t =
  std::enable_if_t<std::decay_t<Arena>::contexpr_size, int>;

/**
 * Defines a valid type if the Arena does not have a contexpr size.
 * \tparam Arena The arena to base the enable on.
 */
template <typename Arena>
using arena_non_constexpr_size_enable_t =
  std::enable_if_t<!std::decay_t<Arena>::contexpr_size, int>;

} // namespace ripple

#endif // RIPPLE_ALLOCATION_ARENA_HPP
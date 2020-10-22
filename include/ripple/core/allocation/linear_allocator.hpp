//==--- ripple/allocation/linear_allocator.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  linear_allocator.hpp
/// \brief This file defines a linear allocator implementation.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALLOCATION_LINEAR_ALLOCATOR_HPP
#define RIPPLE_ALLOCATION_LINEAR_ALLOCATOR_HPP

#include <ripple/core/utility/memory.hpp>
#include <algorithm>

namespace ripple {

/**
 * This allocator allocates data linearly from a provided arena. While it
 * provides an interface for freeing an individual element, it's an empty
 * function, and the allocator only allows resetting all allocations from the
 * pool. It just bumps along the pointer to the next allocation address. It can
 * allocate different sizes.
 */
class LinearAllocator {
 public:
  /**
   * Constructor to set the begin and end of the available memory for the
   * allocator.
   * \param begin The start of the allocation arena.
   * \param end   The end of the allocation arena.
   */
  ripple_host_device LinearAllocator(void* begin, void* end) noexcept
  : begin_(begin), size_(uintptr_t(end) - uintptr_t(begin)) {}

  /**
   * Constructor which takes an Arena which defines the regions from which the
   * allocator can allocate.
   * \param  arena The area to allocate memory from.
   * \tparam Arena The type of the arena.
   */
  template <typename Arena>
  ripple_host_device explicit LinearAllocator(const Arena& arena) noexcept
  : LinearAllocator(arena.begin(), arena.end()) {}

  /** Defstructor -- resets the memory if the pointer is not null */
  ripple_host_device ~LinearAllocator() noexcept {
    if (begin_ != nullptr) {
      begin_   = nullptr;
      current_ = 0;
      size_    = 0;
    }
  };

  /**
   * Move construcor, swaps the other allocator with this one.
   * \param other The other allocator to create this one from.
   */
  ripple_host_device LinearAllocator(LinearAllocator&& other) noexcept {
    swap(other);
  }

  /**
   * Move assignment, swaps the other allocator with this one.
   * \param other The other allocator to swap with this one.
   * \return The newly constructed allocator.
   */
  auto operator=(LinearAllocator&& other) noexcept -> LinearAllocator& {
    if (this != &other) {
      swap(other);
    }
    return *this;
  }

  // clang-format off
  /** Copy constructor -- deleted. */
  LinearAllocator(const LinearAllocator&) = delete;
  /** Copy assignment -- deleted. */
  auto operator=(const LinearAllocator&)  = delete;
  // clang-format on

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Allocates the given number of bytes with the given alignment.
   *
   * \param size      The number of bytes to allocate.
   * \param alignment The alignment for the allocation.
   * \return A valid pointer if the allocation succeeded, otherwise a nullptr.
   */
  ripple_host_device auto
  alloc(size_t size, size_t alignment) noexcept -> void* {
    void* const ptr     = align_ptr(current(), alignment);
    void* const curr    = offset_ptr(ptr, size);
    bool        success = curr <= end();
    set_current(success ? curr : current());
    return success ? ptr : nullptr;
  }

  /**
   * This __does not__ free the \p ptr, since it does not allow freeing of
   * individual allocations. This allocator only allows resetting.
   *
   * \note This is provided so that the allocator has the same interface as
   *       other allocators.
   *
   * \param ptr The pointer to free.
   */
  ripple_host_device auto free(void* ptr) noexcept -> void {}

  /**
   * This __does not__ free the \p ptr, since it does not allow freeing of
   * individual allocations. This allocator only allows resetting.
   *
   * \note This is provided so that the allocator has the same interface as
   *       other allocators.
   *
   * \param ptr  The pointer to free.
   * \param size The size to free.
   */
  ripple_host_device auto free(void* ptr, size_t size) noexcept -> void {}

  /**
   * Gets the amount of allocation space remaining in the allocator.
   * \return The amount of allocation space remaining in the allocator.
   */
  ripple_host_device auto capacity() noexcept -> size_t {
    return uintptr_t(offset_ptr(begin_, size_)) - uintptr_t(current_);
  }

  /**
   * Resets the allocator to the begining of the allocation arena. This
   * invalidates any allocations from the allocator, since any subsequent
   * allocations will overwrite old allocations.
   */
  ripple_host_device auto
  reset(void* begin = nullptr, void* end = nullptr) noexcept -> void {
    current_ = 0;
    if (begin != nullptr && end != nullptr) {
      begin_ = begin;
      size_  = uintptr_t(end) - uintptr_t(begin_);
    }
  }

 private:
  void*    begin_   = nullptr; //!< Pointer to the start of the region.
  uint32_t size_    = 0;       //!< Size of the region.
  uint32_t current_ = 0;       //!< Current allocation location.

  /**
   * Gets a pointer to the current pointer.
   * \return A pointer to the current allocation position.
   */
  ripple_nodiscard ripple_host_device auto current() const noexcept -> void* {
    return offset_ptr(begin_, current_);
  }

  /**
   * Sets the value of pointer for the current allocation such that when
   * offsetting from the beginning of the memory space, it points to the current
   * allocation position.
   * \param current The pointer to set current to point to.
   */
  ripple_host_device auto set_current(void* current) noexcept -> void {
    current_ = uintptr_t(current) - uintptr_t(begin_);
  }

  /**
   * Gets the end of the allocators allocation arena.
   * \return A pointer to the end of the allocation arena.
   */
  ripple_nodiscard ripple_host_device auto end() const noexcept -> void* {
    return offset_ptr(begin_, size_);
  }

  /**
   * Swaps the other allocator with this one.
   * \param other The other allocator to swap with this one.
   */
  ripple_host_device auto swap(LinearAllocator& other) noexcept -> void {
    std::swap(begin_, other.begin_);
    std::swap(size_, other.size_);
    std::swap(current_, other.current_);
  }
};

} // namespace ripple

#endif // RIPPLE_CORE_MEMORY_LINEAR_ALLOCATOR_HPP
//==--- ripple/core/allocation/aligned_heap_allocator.hpp -- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  aligned_heap_allocator.hpp
/// \brief This file defines an allocator which uses aligned_alloc for
///        allocation.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALLOCATION_ALIGNED_HEAP_ALLOCATOR_HPP
#define RIPPLE_ALLOCATION_ALIGNED_HEAP_ALLOCATOR_HPP

#include <cstdlib>

namespace ripple {

/// This type implements an allocator which allocates aligned memory on the
/// heap. It's the slowest allocator, and should therefore always be used as
/// the last resort allocator.
class AlignedHeapAllocator {
 public:
  //==--- [construction] ---------------------------------------------------==//

  // clang-format off
  /// Default constructor.
  AlignedHeapAllocator()  = default;
  /// Destructor -- defaulted.
  ~AlignedHeapAllocator() = default;

  /// Constructor which takes an Arena, which is provided for compatability with
  /// other allocators.
  /// \param  arena The area to allocate memory from. Unused by this allocator.
  /// \tparam Arena The type of the arena.
  template <typename Arena>
  AlignedHeapAllocator(const Arena& arena) {}

  /// Move construcor -- defaulted.
  AlignedHeapAllocator(AlignedHeapAllocator&&)                    = default;
  /// Move assignment -- defaulted.
  auto operator=(AlignedHeapAllocator&&) -> AlignedHeapAllocator& = default;

  //==--- [deleted] --------------------------------------------------------==//

  /// Copy constructor -- deleted since allocators can't be moved.
  AlignedHeapAllocator(const AlignedHeapAllocator&) = delete;
  /// Copy assignment -- deleted since allocators can't be copied.
  auto operator=(const AlignedHeapAllocator&)       = delete;
  // clang-format on

  //==--- [interface] ------------------------------------------------------==//

  /// Allocates \p size bytes of memory with \p align alignment.
  /// \param size  The size of the memory to allocate.
  /// \param align The alignment of the allocation.
  auto alloc(size_t size, size_t alignment) noexcept -> void* {
    return aligned_alloc(alignment, size);
  }

  /// Frees the memory pointed to by ptr.
  /// \param ptr The pointer to the memory to free.
  auto free(void* ptr) noexcept -> void {
    std::free(ptr);
  }

  /// Frees the memory pointed to by \ptr, with a size of \p size.
  /// \param ptr  The pointer to the memory to free.
  auto free(void* ptr, size_t) noexcept -> void {
    std::free(ptr);
  }
};

} // namespace ripple

#endif // RIPPLE_ALLOCATION_ALIGNED_HEAP_ALLOCATOR_HPP
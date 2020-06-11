//==--- ripple/allocation/pool_allocator.hpp --------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  pool_allocator.hpp
/// \brief This file defines a pool allocator implementation.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_ALLOCATION_POOL_ALLOCATOR_HPP
#define RIPPLE_ALLOCATION_POOL_ALLOCATOR_HPP

#include <ripple/core/utility/memory.hpp>
#include <atomic>
#include <cstddef>

namespace ripple {

//==--- [single-threaded free-list] ----------------------------------------==//

/// This type is a simple, single-threaded free-list implementation, which is
/// essentially just a singly-linked-list over an arena from which the free list
/// nodes point to.
class Freelist {
 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor.
  Freelist() = default;

  /// Default destructor.
  ~Freelist() noexcept {
    if (_head != nullptr) {
      _head = nullptr;
    }
  }

  /// Constructor to initialize the freelist with the \p start and \p end of the
  /// arena from which elements can be stored.
  /// \param start        The start of the arena.
  /// \param end          The end of the arena.
  /// \param element_size The size of the elements in the freelist.
  /// \param alignment    The alignment of the elements.
  Freelist(
    const void* start,
    const void* end,
    size_t      element_size,
    size_t      alignment) noexcept
  : _head(initialize(start, end, element_size, alignment)) {}

  // clang-format off
  /// Move constructor to move \p other to this freelist.
  /// \param other The other freelist to move.
  Freelist(Freelist&& other) noexcept                    = default;
  /// Move assignment to move \p other to this freelist.
  /// \param other The other freelist to move.
  auto operator=(Freelist&& other) noexcept -> Freelist& = default;

  //==--- [deleted] --------------------------------------------------------==//

  /// Copy constructor -- deleted since the freelist can't be copied.
  Freelist(const Freelist&)       = delete;
  /// Copy assignment -- deleted since the freelist can't be copied.
  auto operator=(const Freelist&) = delete;
  // clang-format on

  //==--- [interface] ------------------------------------------------------==//

  /// Pops the most recently added element from the list, and returns it.
  auto pop_front() noexcept -> void* {
    Node* const popped_head = _head;
    _head                   = popped_head ? popped_head->next : nullptr;
    return static_cast<void*>(popped_head);
  }

  /// Pushes a new element onto the front of the list.
  /// \param ptr The pointer to the element to push onto the list.
  auto push_front(void* ptr) noexcept -> void {
    if (ptr == nullptr) {
      return;
    }

    Node* const pushed_head = static_cast<Node*>(ptr);
    pushed_head->next       = _head;
    _head                   = pushed_head;
  }

 private:
  /// Simple node type which points to the next element in the list.
  struct Node {
    Node* next = nullptr; //!< The next node in the list.
  };

  Node* _head = nullptr; //!< Pointer to the head of the list.

  /// Intializes the free list by building the linked list.
  /// \param start        The start of the freelist arena.
  /// \param end          The end of the freelist arena.
  /// \param element_size The size of the elements in the freelist.
  /// \param alignment    The alignment of the elements.
  static auto initialize(
    const void* start, const void* end, size_t element_size, size_t alignment)
    -> Node* {
    // Create the first and second elements:
    void* const first  = align_ptr(start, alignment);
    void* const second = align_ptr(offset_ptr(first, element_size), alignment);

    const size_t size     = uintptr_t(second) - uintptr_t(first);
    const size_t elements = (uintptr_t(end) - uintptr_t(first)) / size;

    // Set the head of the list:
    Node* head = static_cast<Node*>(first);

    // Initialize the rest of the list:
    Node* current = head;
    for (size_t i = 1; i < elements; ++i) {
      Node* next    = static_cast<Node*>(offset_ptr(current, size));
      current->next = next;
      current       = next;
    }
    assert(
      offset_ptr(current, size) <= end &&
      "Freelist initialization overflows provided arena!");

    current->next = nullptr;
    return head;
  }
};

//==--- [multi-threaded free list] -----------------------------------------==//

/// This type is a thread-safe freelist implementation. It's lock-free, and uses
/// atomics to control access to the nodes in the freelist.
///
/// Use this only when data needs to be shared across threads, and even so,
/// compare the performance against other allocators with a locking policy, such
/// as the single-threaded Freelist with a spinlock.
///
/// __Benchmark this against other options.__
///
/// Also consider using thread-local freelists with a common arena, with this as
/// a fallback.
class ThreadSafeFreelist {
  /// The next pointer for the node is atomic because the thread sanitizer
  /// says that there is a data race for the following situation:
  ///
  /// When pop_front starts in one thread and reads the next pointer from the
  /// head, but another thread does also does a pop_front, completes first, and
  /// therefore moves the _head. The other thread then does a push_front, and
  /// both threads will be trying to perform the compare_exchange. However,
  /// because of the tagging, the pop thread will have an invalid tag, so the
  /// compare_exchange will fail for the pop thread, but will succeed for the
  /// pushing thread, since it has the correct head. The failed popping thread
  /// will then try again, and everything is fine. It looks as follows:
  ///
  ///     Thread 1                   Thread 2
  ///        |                          |
  ///     pop_front()                   |
  ///        |                          |
  ///   read head->next                 |
  ///        |                      pop_front()
  ///        |                          |
  ///        |                     read head->next
  ///        |                          |
  ///        |             compare_exchange success, tag++
  ///        |                          |
  ///        |                     push_front()
  ///        |                          |
  ///   data race on               data race on
  ///  write head->next           write head->next
  ///        |                          |
  /// compare_exchange           compare_exhange
  ///   always fails           always succeeds, tag++
  ///        |                          |
  ///        |                        return
  ///     try again                     |
  ///   read head->next                 |
  ///        |                          |
  ///   compare_exchange                |
  ///   succeeds, tag++                 |
  ///
  /// While this doesn't cause any problems, we just make the next pointer
  /// atomic and use it with relaxed ordering, which doesn't reduce performance
  /// or increase the storage requirement.
  struct Node {
    std::atomic<Node*> next; //!< Pointer to the next node.
  };

  /// Defines the alignement for the head pointer.
  static constexpr size_t head_ptr_alignment_bytes = 8;

  /// This struct is using a 32-bit offset into the freelist arena rather than
  /// a direct pointer, because together with the 32-bit tag, it needs to
  /// fit into 8 bytes to be atomic, which would not be the case if the offset
  /// was a pointer (even if the range of the tag was limited.
  ///
  /// The tag is required so that there is no ABA problem where there is a pop
  /// in one thread, and a pop -> push in another, causing the new pushed head
  /// in the one thread to be in the same place as the popped head in the other,
  /// but with potentially different data. Updating the tag prevents this.
  ///
  /// See the description in Node.
  struct alignas(head_ptr_alignment_bytes) HeadPtr {
    int32_t  offset; //!< Offset into the arena, rather than pointer.
    uint32_t tag;    //!< Tag to ensure atomic operations are correct.
  };

  /// Defines the type of an atomic head pointer.
  using atomic_head_ptr_t = std::atomic<HeadPtr>;

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor.
  ThreadSafeFreelist() noexcept = default;

  /// Constructor to initialize the freelist with the \p start and \p end of the
  /// arena from which elements can be stored.
  /// \param start        The start of the arena.
  /// \param end          The end of the arena.
  /// \param element_size The size of the elements in the freelist.
  /// \param alignment    The alignment of the elements.
  ThreadSafeFreelist(
    const void* start,
    const void* end,
    size_t      element_size,
    size_t      alignment) noexcept {
    assert(_head.is_lock_free());

    void* const first  = align_ptr(start, alignment);
    void* const second = align_ptr(offset_ptr(first, element_size), alignment);

    // Check that the resulting pointers are in the arena, and ordered
    // correctly:
    assert(first >= start && first < end);
    assert(second >= start && second > first && second < end);

    const size_t size     = uintptr_t(second) - uintptr_t(first);
    const size_t elements = (uintptr_t(end) - uintptr_t(first)) / size;

    // Set the head to the first element, and the storage to the head.
    Node* head = static_cast<Node*>(first);
    _storage   = head;

    // Link the list:
    Node* current = head;
    for (size_t i = 1; i < elements; ++i) {
      Node* next    = static_cast<Node*>(offset_ptr(current, size));
      current->next = next;
      current       = next;
    }

    // Ensure that everything fits in the allocation arena.
    assert(current < end);
    assert(offset_ptr(current, size) <= end);
    current->next = nullptr;

    // Set the head pointer as the offset from the storage to the aligned head
    // element, and set the initial tag to zero.
    _head.store({static_cast<int32_t>(head - _storage), 0});
  }

  /// Move constructor to move \p other to this freelist.
  /// \param other The other freelist to move.
  ThreadSafeFreelist(ThreadSafeFreelist&& other) noexcept
  : _head(other._head.load(std::memory_order_relaxed)),
    _storage(std::move(other._storage)) {
    other._head.store({-1, 0}, std::memory_order_relaxed);
    other._storage = nullptr;
  }

  /// Move assignment to move \p other to this freelist.
  /// \param other The other freelist to move.
  auto operator=(ThreadSafeFreelist&& other) noexcept -> ThreadSafeFreelist& {
    if (this != &other) {
      _head.store(
        other._head.load(std::memory_order_relaxed), std::memory_order_relaxed);
      _storage = std::move(other._storage);
      other._head.store({-1, 0}, std::memory_order_relaxed);
      other._storage = nullptr;
    }
    return *this;
  }

  //==--- [deleted] --------------------------------------------------------==//

  //clang-format off
  /// Copy constructor -- deleted since the freelist can't be copied.
  ThreadSafeFreelist(const ThreadSafeFreelist&) = delete;
  /// Copy assignment -- deleted since the freelist can't be copied.
  auto operator=(const ThreadSafeFreelist&) = delete;
  // clang-format on

  //==--- [interface] ------------------------------------------------------==//

  /// Pops the most recently added element from the list, and returns it. If
  /// there are no elements left in the list, this returns a nullptr.
  auto pop_front() noexcept -> void* {
    Node* const storage = _storage;

    // Here we acquire to synchronize with other popping threads which may
    // succeed first, and well as with other pushing threads which may push
    // before we pop, in which case we want to try and pop the newly pushed
    // head.
    HeadPtr current_head = _head.load(std::memory_order_acquire);

    while (current_head.offset >= 0) {
      // If another thread tries to pop, and does it faster than here, then this
      // pointer will contain data from the application. However, the new_head
      // which we compute just now, using this next pointer, will be discarded
      // because _head will have been replaced, and hence current_head will not
      // compare equal with _head, and thus the compare_exhange will fail and we
      // will try again.
      Node* const next =
        storage[current_head.offset].next.load(std::memory_order_relaxed);

      // Get the new head element. If the next pointer is a nullptr, then we are
      // at the end of the list, and if we succeed then another thread cannot
      // pop, so we set the offset to -1 so that on success, if _head is
      // replaced with new_head, then other threads will not execute this loop
      // and just return a nullptr.
      const HeadPtr new_head{
        next ? int32_t(next - storage) : -1, current_head.tag + 1};

      // If another thread was trying to pop, and got here just before us, then
      // the _head element would have moved, and it will have a different .tag
      // value than the tag value of current_head, so this will fail, and we
      // will try again.
      //
      // This is also how we avoid the ABA problem, where another thread might
      // have popped, and then pushed, leaving the head in the same place as the
      // current_head we now have, but with different data. The tag will prevent
      // this.
      //
      // Also, we need memory_order_release here to make the change visible on
      // success, for threads which load the upated _head. We use
      // memory_order_acquire for the failure case, because we want to
      // synchronize as at the beginning of the function.
      if (_head.compare_exchange_weak(
            current_head,
            new_head,
            std::memory_order_release,
            std::memory_order_acquire)) {
        // Lastly, we need the assert here for that case that another thread
        // performed a pop between our load of _head and _head.next. In this
        // case, the next that we loaded will contain application data, and
        // therefore could be invalid. So we check that we either have a
        // nullptr, in the case that we are at the last element, or that the
        // next pointer is in the memory arena, otherwise something went wrong.
        assert(!next || next >= storage);
        break;
      }
    }

    // Either we have the head, and we can return it, or we ran out of elements,
    // and we have to return a nullptr.
    // clang-format off
    void* p = (current_head.offset >= 0) 
            ? storage + current_head.offset : nullptr;
    // clang-format on
    return p;
  }

  /// Pushes the \ptr onto the front of the free list.
  /// \param ptr The pointer to push onto the front.
  auto push_front(void* ptr) noexcept -> void {
    Node* const storage = _storage;
    assert(ptr && ptr >= storage);
    Node* const node = static_cast<Node*>(ptr);

    // Here we don't care about synchronization with stores to _head from other
    // threads which are either trying to push or to pop. If that happens, the
    // compare exchange will fail and we will just try with the newly updated
    // head.
    HeadPtr current_head = _head.load(std::memory_order_relaxed);
    HeadPtr new_head     = {int32_t(node - storage), current_head.tag + 1};

    // Here we use memory_order_release in the success case, so that other
    // threads can synchronize with the updated head, but we don't care about
    // synchronizing the update of the current head because as above, if another
    // thread races ahead, we will just try again. The memory ordering with
    // respect to the current_head update is not important.
    do {
      // clang-format off
      new_head.tag     = current_head.tag + 1;
      Node* const next = (current_head.offset >= 0)
                       ? (storage + current_head.offset) : nullptr;
      node->next.store(next, std::memory_order_relaxed);
      // clang-format on
    } while (!_head.compare_exchange_weak(
      current_head,
      new_head,
      std::memory_order_release,
      std::memory_order_relaxed));
  }

 private:
  atomic_head_ptr_t _head{};            //!< Head pointer (index).
  Node*             _storage = nullptr; //!< Storage.
};

//==--- [pool allocator] ---------------------------------------------------==//

/// Allocator to allocator elements of size ElementSize, with a given Alignment,
/// and using the given FreelistImpl.
/// We don't use a templated type for the pooled allocator to allow the
/// allocator to allocate elements of different types but which are the same
/// size, or smaller, than the element size for the pool.
///
/// \tparam ElementSize  The byte size of the elements in the pool.
/// \tparam Alignment    The alignment for the elements.
/// \tparam FreelistImpl The implementation type of the freelist.
template <
  size_t ElementSize,
  size_t Alignment,
  typename FreelistImpl = Freelist>
class PoolAllocator {
 private:
  // clang-format off
  /// Defines the size of the pool elements.
  static constexpr size_t element_size = ElementSize;
  /// Defines the alignment of the allocations.
  static constexpr size_t alignment    = Alignment;

  /// Defines the type of the freelist.
  using freelist_t = FreelistImpl;
  //clang-format on

 public:
  //==--- [construction] ---------------------------------------------------==//

  // clang-format off
  /// Default constructor for the pool.
  PoolAllocator() noexcept  = delete;
  /// Default destructor for the pool.
  ~PoolAllocator() noexcept = default;
  // clang-format on

  /// Constructor which initializes the freelist with the \p start and \p end
  /// pointers to the memory arena for the pool.
  /// \param start A pointer to the start of the arena for the pool.
  /// \param end   A pointer to the end of the arena for the pool.
  PoolAllocator(const void* begin, const void* end) noexcept
  : _freelist(begin, end, element_size, alignment), _begin(begin), _end(end) {}

  /// Constructor to initialize the allocator with the arena to allocator from.
  /// \param  arena The arena for allocation.
  /// \tparam Arena The type of the arena.
  template <typename Arena>
  explicit PoolAllocator(const Arena& arena) noexcept
  : PoolAllocator(arena.begin(), arena.end()) {}

  /// Moves constructor to move \p other into this allocator.
  /// \param other The other allocator to move into this one.
  PoolAllocator(PoolAllocator&& other) noexcept = default;

  /// Move assignment operator to move \p other into this allocator.
  /// \param other The other allocator to move into this one.
  auto operator=(PoolAllocator&& other) noexcept -> PoolAllocator& = default;

  //==--- [deleted] --------------------------------------------------------==//

  // clang-format off
  /// Copy constructor -- deleted, allocators can't be copied.
  PoolAllocator(const PoolAllocator&)  = delete;
  /// Copy assignment -- deleted, allocators can't be copied.
  auto operator=(const PoolAllocator&) = delete;
  // clang-format on

  //==--- [interface] ------------------------------------------------------==//

  /// Allocates an element of \p size with a given \p alignment from the pool.
  /// This will fail if \p size is larger than the element size for the pool or
  /// if the alignment is larger than the alignment for the pool.
  ///
  /// If the pool is full, this will return a nullptr.
  ///
  /// \param size  The size of the element to allocate.
  /// \param align The alignment for the allocation.
  auto alloc(size_t size = element_size, size_t align = alignment) noexcept
    -> void* {
    assert(size <= element_size && align <= alignment);
    return _freelist.pop_front();
  }

  /// Frees the \p ptr, pushing it onto the front of the freelist.
  /// \param ptr The pointer to free.
  auto free(void* ptr, size_t = element_size) noexcept -> void {
    _freelist.push_front(ptr);
  }

  /// Returns true if the allocator owns the \p ptr.
  /// \param ptr The pointer to determine if is owned by the allocator.
  auto owns(void* ptr) const noexcept -> bool {
    return uintptr_t(ptr) >= uintptr_t(_begin) &&
           uintptr_t(ptr) < uintptr_t(_end);
  }

  /// Resets the pool. Since the freelist doesn't support restting, this doesn't
  /// do anything.
  auto reset() noexcept -> void {}

 private:
  freelist_t  _freelist; //!< The freelist to allocate from and free into.
  const void* _begin = nullptr; //!< The beginning of the pool arena.
  const void* _end   = nullptr; //!< The end of the pool arena.
};

} // namespace ripple

#endif // RIPPLE_ALLOCATION_POOL_ALLOCATOR_HPP
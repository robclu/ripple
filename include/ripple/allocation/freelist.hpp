/**=--- ripple/allocation/freelist.hpp --------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  freelist.hpp
 * \brief This file defines a simple freelist class.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_ALLOCATION_FREELIST_HPP
#define RIPPLE_ALLOCATION_FREELIST_HPP

#include <ripple/utility/memory.hpp>

namespace ripple {

/**
 * This type is a simple, single-threaded freelist implementation, which is
 * essentially just a singly-linked-list over an arena from which the free list
 * nodes point to.
 *
 * This can be used in both host and device code, so can quickly allocate
 * dynamically on the device.
 */
class Freelist {
  /**
   * Simple node type which points to the next element in the list.
   */
  struct Node {
    Node* next = nullptr; //!< The next node in the list.
  };

 public:
  /** Default constructor. */
  ripple_host_device Freelist() noexcept : head_{nullptr} {}

  /**
   * Destructor, resets the head pointer if it's not a nullptr.
   */
  ripple_host_device ~Freelist() noexcept {
    if (head_ != nullptr) {
      head_ = nullptr;
    }
  }

  /**
   * Constructor to initialize the freelist with the start and end of the
   * arena from which elements can be stored.
   * \param start        The start of the arena.
   * \param end          The end of the arena.
   * \param element_size The size of the elements in the freelist.
   * \param alignment    The alignment of the elements.
   */
  ripple_host_device Freelist(
    const void* start,
    const void* end,
    size_t      element_size,
    size_t      alignment) noexcept
  : head_{initialize(start, end, element_size, alignment)} {}

  /**
   * Move constructor to move the other freelist to this one.
   * \param other The other freelist to move.
   */
  Freelist(Freelist&& other) noexcept = default;

  /**
   * Move assignment to move the other freelist to this one.
   * \param other The other freelist to move.
   */
  auto operator=(Freelist&& other) noexcept -> Freelist& = default;

  // clang-format off
  /** Copy constructor -- deleted since the freelist can't be copied. */
  Freelist(const Freelist&)       = delete;
  /** Copy assignment -- deleted since the freelist can't be copied. */
  auto operator=(const Freelist&) = delete;
  // clang-format on

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Pops the most recently added element from the list, and returns a pointer
   * to it.
   * \return A pointer to the most recently added element.
   */
  ripple_host_device auto pop_front() noexcept -> void* {
    Node* const popped_head = head_;
    head_                   = popped_head ? popped_head->next : nullptr;
    return static_cast<void*>(popped_head);
  }

  /**
   * Pushes a new element onto the front of the list.
   * \param ptr The pointer to the element to push onto the list.
   */
  ripple_host_device auto push_front(void* ptr) noexcept -> void {
    if (ptr == nullptr) {
      return;
    }

    Node* const pushed_head = static_cast<Node*>(ptr);
    pushed_head->next       = head_;
    head_                   = pushed_head;
  }

 private:
  Node* head_ = nullptr; //!< Pointer to the head of the list.

  /**
   * Initializes the freelist by linking the nodes.
   * \param start        The start of the freelist arena.
   * \param end          The end of the freelist arena.
   * \param element_size The size of the elements in the freelist.
   * \param alignment    The alignment of the elements.
   */
  ripple_host_device static auto initialize(
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

} // namespace ripple

#endif // RIPPLE_ALLOCATION_FREELIST_HPP
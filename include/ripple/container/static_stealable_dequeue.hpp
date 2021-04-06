/**=--- ripple/container/static_stealable_dequeue.hpp ------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  static_stealable_dequeue.hpp
 * \brief This file defines an implemenation of a static stealable double
 *        ended queue. The queue is lock free.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_CONTAINER_STATIC_STEALABLE_DEQUEUE_HPP
#define RIPPLE_CONTAINER_STATIC_STEALABLE_DEQUEUE_HPP

#include <array>
#include <atomic>
#include <optional>

namespace ripple {

/**
 * The StaticStealableDequeue is a fixed-size, concurrent, lock-free stealable
 * double-ended queue implementation. It is designed for a single thread to
 * push and pop onto and from the bottom of the queue, respectively, and any
 * number of threads to steal from the top of the queue.
 *
 * The API provides by try_push, push, pop, steal, and size.
 *
 * The push function will push an object onto the queue regardless of whether
 * the queue is full or not, and since the queue is circular, will therefore
 * overwrite the oldest object which was pushed onto the queue. It is faster
 * than try_push, but must only be used when it's known that the queue is not
 * full. In debug mode, push will notify if an overwrite happens.
 *
 * The try_push function calls push if the queue is not full.
 *
 * pop and steal will return a nullopt if the queue is empty, or if another
 * thread stole the last element before they got to it. Otherwise they will
 * return a pointer to the object.
 *
 *  To provide an API that does not easily introduce bugs, all returned objects
 *  are __copies__, thus the queue is not suitable for large objects. This is
 *  because providing a reference or pointer to an element introduces a
 *  potential race between the use of that element and a push to the queue.
 *
 *  Consider if the queue is full. Calling ``try_push`` returns false, as one
 *  would expect, and calling ``steal`` returns the oldest element (say at
 *  index 0) then increments top (to 1). Another call to ``try_push`` then
 *  succeeds because the queue is not at capacity, but it creates the new
 *  element in the same memory location as the element just stolen. Therefore,
 *  if a pointer is returned, there is a race between the stealing thread to use
 *  the element pointed to, and the pushing thread overwriting the memory. Such
 *  a scenario is too tricky to use correctly to include in the API. Therefore,
 *  benchmark the queue for the required data type, or run in debug mode to
 *  determine a good size for the queue. This queue is not dynamically sized to
 *  improve performance.
 *
 *  Also note that due to the multi-threaded nature of the queue, a call to
 *  size only returns the size of the queue at that point in time, and it's
 *  possible that other threads may have stolen from the queue between the time
 *  at which size was called and then next operation.
 *
 *  This imlementation is based on the original paper by Chase and Lev:
 *
 *    [Dynamic circular work-stealing deque](
 *      http://dl.acm.org/citation.cfm?id=1073974)
 *
 *  without the dynamic resizing.
 *
 *  __Note:__ This class is aligned to avoid false sharing. Even though the
 *  objects in the queue are stored on the stack, it's possible that the last
 *  cache line isn't full, so we need to avoid other members of the class being
 *  put on the cacheline.
 *
 *  ~~~cpp
 *  struct alignas(avoid_false_sharing_size) SomeClass {
 *    StaticStealableQueue<SomeType, 1024> queue;
 *  };
 *  ~~~
 *
 *  \tparam Type        The type of the data in the queue.
 *  \tparam MaxElements The maximum number of elements for the queue.
 */
template <typename Type, uint32_t MaxElements>
class alignas(avoid_false_sharing_size) StaticStealableDequeue {
 public:
  // clang-format off
  /** Defines the size type of for the indices. */
  using Size      = uint32_t;
  /** Defines the atomic type used for the index pointers. */
  using Atomic    = std::atomic<Size>;
  /** Defines the type of the elements in the queue. */
  using Element   = Type;
  /** Defines an optional typeto use when returning value types. */
  using Optional  = std::optional<Element>;
  /** Defines the type of container used to store the queue's objects. */
  using Container = std::vector<Element>;
  // clang-format on

 private:
  /** Defines a valid type if the object is a pointer. */
  template <typename... Args>
  using pointer_enable_t =
    std::enable_if_t<(std::is_pointer_v<Args> * ... * true), int>;

  /** Defines a valid type if the object is not a pointer. */
  template <typename... Args>
  using non_pointer_enable_t =
    std::enable_if_t<(!std::is_pointer_v<Args> * ... * true), int>;

 public:
  /** Default constructor for the queue. */
  StaticStealableDequeue() = default;

  /**
   * Move constructor to move the  other queue into this one.
   * \param other The other queue to move into this one.
   */
  StaticStealableDequeue(StaticStealableDequeue&& other) noexcept
  : elements_{ripple_move(other.elements_)},
    top_{other.top_.load()},
    bottom_{other.bottom_.load()} {}

  /** Copy constructor -- deleted. */
  StaticStealableDequeue(const StaticStealableDequeue&) = delete;

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Pushes an element onto the front of the queue, when the element is an
   * rvalue reference type.
   *
   * \note This does not check if the queue is full, and hence it
   *       __will overwrite__ the least recently added element if the queue is
   *       full
   *
   * \param  args  The arguments used to construct the object.
   * \tparam Args  The type of the arguments for object construction.
   */
  template <typename T, pointer_enable_t<T> = 0>
  auto push(T ptr) noexcept -> void {
    const auto bottom = bottom_.load(std::memory_order_relaxed);
    const auto index  = wrapped_index(bottom);

    elements_[index] = ptr;

    // Ensure that the compiler does not reorder this instruction and the
    // setting of the object above, otherwise the object seems added (since the
    // bottom index has moved) but isn't (since it would not have been added).
    bottom_.store(bottom + 1, std::memory_order_release);
  }

  /**
   * Pushes an element onto the front of the queue, when the element is an
   * rvalue reference type.
   *
   * \note This does not check if the queue is full, and hence it
   *       __will overwrite__ the least recently added element if the queue is
   *      full.
   *
   * \param  args  The arguments used to construct the object.
   * \tparam Args  The type of the arguments for object construction.
   */
  template <typename... Args, non_pointer_enable_t<Args...> = 0>
  auto push(Args&&... args) noexcept -> void {
    const auto bottom = bottom_.load(std::memory_order_relaxed);
    const auto index  = wrapped_index(bottom);

    new (&elements_[index]) Element(ripple_forward(args)...);

    // Ensure that the compiler does not reorder this instruction and the
    // setting of the object above, otherwise the object seems added (since the
    // bottom index has moved) but isn't (since it would not have been added).
    bottom_.store(bottom + 1, std::memory_order_release);
  }

  /**
   * Tries to push an object onto the front of the queue.
   * \param  args The arguments used to construct the object.
   * \tparam Args The type of the arguments for object construction.
   * \return true if a new object was pushed onto the queue.
   */
  template <typename... Args>
  auto try_push(Args&&... args) noexcept -> bool {
    if (size() >= MaxElements) {
      return false;
    }

    push(ripple_forward(args)...);
    return true;
  }

  /**
   * Pops an object from the front of the queue. This returns an optional
   * type which holds either the object at the bottom of the queue (the most
   * recently added object) or an invalid optional.
   *
   * ~~~cpp
   * // If using the object directly:
   * if (auto object = queue.pop()) {
   *   object->invoke_oject_member_function();
   * }
   *
   * // If using the object for a call:
   * if (auto object = queue.pop()) {
   *   // Passes object by reference:
   *   function_using_bject(*object);
   * }
   * ~~~
   *
   * \return An optional type which is in a valid state if the queue is not
   *         empty.
   */
  auto pop() noexcept -> Optional {
    auto bottom = bottom_.load(std::memory_order_relaxed) - 1;

    // Sequentially consistant memory ordering is used here to ensure that the
    // load to top always happens after the load to bottom above, and that the
    // compiler emits an __mfence__ instruction.
    bottom_.store(bottom, std::memory_order_seq_cst);

    auto top = top_.load(std::memory_order_relaxed);
    if (top > bottom) {
      bottom_.store(top, std::memory_order_relaxed);
      return std::nullopt;
    }

    auto object = std::make_optional(elements_[wrapped_index(bottom)]);
    if (top != bottom) {
      return object;
    }

    // If we are here there is only one element left in the queue, and there may
    // be a race between this method and steal() to get it. If this exchange is
    // true then this method won the race (or was not in one) and then the
    // object can be returned, otherwise the queue has already been emptied.
    bool exchanged =
      top_.compare_exchange_strong(top, top + 1, std::memory_order_release);

    // This is also a little tricky: If we lost the race, top will be changed to
    // the new value set by the stealing thread (i.e it's already incremented).
    // If it's incremented again then bottom_ > top_ when the last item was
    // actually just cleared. This is also the unlikely case -- since this path
    // is only executed when there is contention on the last element -- so the
    // const of the branch is acceptable.
    bottom_.store(top + (exchanged ? 1 : 0), std::memory_order_relaxed);

    return exchanged ? object : std::nullopt;
  }

  /**
   * Steals an object from the top of the queue. This returns an optional type
   * which is the object if the steal was successful, or a default constructed
   * optional if not.
   *
   * Example usage is:
   *
   * ~~~cpp
   *  // If using the object directly:
   *  if (auto object = queue.steal()) {
   *    object->invoke_object_member_function();
   *  }
   *
   *  // If using the object for a call:
   *  if (auto object = queue.steal()) {
   *    // Passes object by reference:
   *    function_using_object(*object);
   *  }
   * ~~~
   *
   * \return A pointer to the top (oldest) element in the queue, or nullptr.
   */
  auto steal() noexcept -> Optional {
    auto top = top_.load(std::memory_order_relaxed);

    // Top must always be set before bottom, so that bottom - top represents an
    // accurate enough (to prevent error) view of the queue size. Loads to
    // different address aren't reordered (i.e load load barrier)
    asm volatile("" ::: "memory");

    auto bottom = bottom_.load(std::memory_order_relaxed);
    if (top >= bottom) {
      return std::nullopt;
    }

    // Here the object at top is fetched, and and update to top_ is atempted,
    // since the top element is being stolen. __If__ the exchange succeeds,
    // then this method won a race with another thread to steal the element,
    // or if there is only a single element left, then it won a race between
    // other threads and the pop() method (potentially), or there was no race.
    // In summary, if the exchange succeeds, the object can be returned,
    // otherwise it can't.
    //
    // Also note that the object __must__ be created before the exchange to top,
    // otherwise there will potentially be a race to construct the object and
    // between a thread pushing onto the queue.
    //
    // This introduces overhead when there is contention to steal, and the steal
    // is unsuccessful, but in the succesful case there is no overhead.
    auto object = std::make_optional(elements_[wrapped_index(top)]);
    bool exchanged =
      top_.compare_exchange_strong(top, top + 1, std::memory_order_release);

    return exchanged ? object : std::nullopt;
  }

  /**
   * Gets the number of elements in the queue. This __does not__ always
   * return the actual size, but an approximation of the size since
   * top can be modified by another thread which steals.
   */
  auto size() const noexcept -> Size {
    return bottom_.load(std::memory_order_relaxed) -
           top_.load(std::memory_order_relaxed);
  }

 private:
  /* Note: The starting values are 1 since the element popping pops (bottom - 1)
   *       so if bottom = 0 to start, then (0 - 1) = size_t_max, and the pop
   *       function tries to access an out of range element. */

  Container elements_{MaxElements}; //!< Container of tasks.
  Atomic    top_    = 1;            //!< The index of the top element.
  Atomic    bottom_ = 1;            //!< The index of the bottom element.

  /**
   * Gets the wrapped index for either the top or bottom of the queue.
   * \param index The index to wrap.
   */
  auto wrapped_index(Size index) const noexcept -> Size {
    return index % MaxElements;
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_STATIC_STEALABLE_DEQUEUE_HPP
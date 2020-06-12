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

#include <ripple/core/utility/memory.hpp>
#include <type_traits>

namespace ripple {

/// Defines a stack-based memory arena of a specific size.
/// \tparam Size The size of the stack for the arena.
template <size_t Size>
class StackArena {
  /// Defines the size of the stack for the arena.
  static constexpr size_t stack_size = Size;

 public:
  //==--- [traits] ---------------------------------------------------------==//

  /// Returns that the allocator has a constexpr size.
  static constexpr bool contexpr_size = true;

  using ptr_t       = void*;       //!< Pointer type.
  using const_ptr_t = const void*; //!< Const pointer type.

  //==--- [constructor] ----------------------------------------------------==//

  /// Constructor which takes the size of the arena. This is provided to arena's
  /// have the same interface.
  /// \param size The size of the arena.
  StackArena(size_t size = 0) {}

  //==--- [interface] ------------------------------------------------------==//

  /// Returns a pointer to the beginning of the arena.
  [[nodiscard]] auto begin() const -> const_ptr_t {
    return static_cast<const_ptr_t>(&_buffer[0]);
  }

  /// Returns a pointer to the end of the arena.
  [[nodiscard]] auto end() const -> const_ptr_t {
    return static_cast<const_ptr_t>(&_buffer[stack_size]);
  }

  /// Returns the size of the arena.
  [[nodiscard]] constexpr auto size() const -> size_t {
    return stack_size;
  }

 private:
  char _buffer[stack_size]; //!< The buffer for the stack.
};

/// Defines a heap-based arena.
struct HeapArena {
 public:
  //==--- [traits] ---------------------------------------------------------==//

  /// Returns that the allocator does not have a constexpr size.
  static constexpr bool constexpr_size = false;

  using ptr_t       = void*; //!< Pointer type.
  using const_ptr_t = void*; //!< Const pointer type.

  //==--- [construction] ---------------------------------------------------==//

  /// Initializes the arena with a specific size.
  /// \param size The size of the arena.
  explicit HeapArena(size_t size) {
    if (size) {
      _start = malloc(size);
      _end   = offset_ptr(_start, size);
    }
  }

  /// Destructor to release the memory.
  ~HeapArena() noexcept {
    if (_start != nullptr) {
      std::free(_start);
      _start = nullptr;
      _end   = nullptr;
    }
  }

  //==--- [deleted] --------------------------------------------------------==//

  // clang-format off
  /// Copy constructor -- deleted.
  HeapArena(const HeapArena&)     = delete;
  /// Move constructor -- deleted.
  HeapArena(HeapArena&&) noexcept = delete;

  /// Copy assignment operator -- deleted.
  auto operator=(const HeapArena&)     = delete;
  /// Move assignment operator -- deleted.
  auto operator=(HeapArena&&) noexcept = delete;
  // clang-format on

  //==--- [interface] ------------------------------------------------------==//

  /// Returns a pointer to the beginning of the arena.
  [[nodiscard]] auto begin() const -> const_ptr_t {
    return _start;
  }

  /// Returns a pointer to the end of the arena.
  [[nodiscard]] auto end() const -> const_ptr_t {
    return _end;
  }

  /// Returns the size of the arena.
  [[nodiscard]] auto size() const -> size_t {
    return uintptr_t(_end) - uintptr_t(_start);
  }

 private:
  void* _start = nullptr; //!< Pointer to the heap data.
  void* _end   = nullptr; //!< Pointer to the heap data.
};

//==--- [aliases] ----------------------------------------------------------==//

/// Defines the default size for a stack arena.
static constexpr size_t default_stack_arena_size = 1024;

/// Defines the type for a default stack arena.
using DefaultStackArena = StackArena<default_stack_arena_size>;

/// Defines a valid type if the Arena has a contexpr size.
/// \tparam Arena The arena to base the enable on.
template <typename Arena>
using arena_constexpr_size_enable_t =
  std::enable_if_t<std::decay_t<Arena>::contexpr_size, int>;

/// Defines a valid type if the Arena does not have a contexpr size.
/// \tparam Arena The arena to base the enable on.
template <typename Arena>
using arena_non_constexpr_size_enable_t =
  std::enable_if_t<!std::decay_t<Arena>::contexpr_size, int>;

} // namespace ripple

#endif // RIPPLE_ALLOCATION_ARENA_HPP
//==--- ripple/core/graph/modifier.hpp --------------------- -*- C++ -*- ---==//
//
//                                  Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  modifier.hpp
/// \brief This file defines a modifier class which is essentially a wrapped
///        which allows the specification for how data is modified.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_GRAPH_MODIFIER_HPP
#define RIPPLE_GRAPH_MODIFIER_HPP

#include <ripple/core/utility/type_traits.hpp>
#include <ripple/core/container/block_traits.hpp>
#include <ripple/core/container/tensor.hpp>

namespace ripple {

/// Defines the type of the modifier.
enum class Modifier : uint8_t {
  readonly        = 0, //!< The modification is readonly.
  writable        = 1, //!< The modification writes to the data.
  shared          = 2, //!< The modifier uses shared memory.
  readonly_shared = 3, //!< The modifier is readonly and uses shared memory.
  writable_shared = 4  //!< The modifier is writable and uses shared memory.
};

/// The Modifier struct wraps a type T with some information for how it is
/// modified.
/// \tparam T            The type which is wrapped in the modifier.
/// \tparam Modification The type of the modification.
template <typename T, Modifier Modification = Modifier::readonly>
struct ModificationSpecifier {
  T wrapped; //!< The type being wrapped.
};

//==--- [traits] -----------------------------------------------------------==//

/// Modification traits for a type which is not a modifier.
/// \tparam T The type to get the traits for.
template <typename T>
struct ModificationTraits {
  /// Defines the wrapped type for the modifier.
  using value_t = remove_ref_t<T>;

  // clang-format off
  /// Returns true if T is a modifier type.
  static constexpr bool is_modifier         = false;
  /// Returns true if the modification is read only.
  static constexpr bool is_readonly         = true;
  /// Returns true if the modification is writable.
  static constexpr bool is_writable         = !is_readonly;
  /// Returns true if the modification requires shared memory.
  static constexpr bool uses_shared         = false;
  /// Returns true if the modifier is readable or writable.
  static constexpr bool read_write_modifier = false;
  // clang-format on
};

/// Modification traits for a type which is a modifier.
/// \tparam T            The type wrapped by the modifier.
/// \tparam Modification The type of modification for the modifier.
template <typename T, Modifier Modification>
struct ModificationTraits<ModificationSpecifier<T, Modification>> {
  /// Defines the wrapped type for the modifier.
  using value_t = remove_ref_t<T>;

  // clang-format off
  /// Returns true if T is a modifier type
  static constexpr bool is_modifier = true;
  /// Returns true if the modification is read only.
  static constexpr bool is_readonly = Modification == Modifier::readonly ||
                                      Modification == Modifier::readonly_shared;
  /// Returns true if the modification is writable.
  static constexpr bool is_writable = Modification == Modifier::writable ||
                                      Modification == Modifier::writable_shared;
  /// Returns true if the modification requires shared memory.
  static constexpr bool uses_shared =
    Modification == Modifier::shared          ||
    Modification == Modifier::writable_shared ||
    Modification == Modifier::readonly_shared;

  /// Returns true if the modification is readable or writable.
  static constexpr bool read_write_modifier = is_readonly || is_writable;
  // clang-format on
};

/// Alias for the modification traits for T.
/// \tparam T The type to get the modification traits for.
template <typename T>
using modification_traits_t = ModificationTraits<std::decay_t<T>>;

//==--- [methods] ----------------------------------------------------------==//

/// Creates a wrapped \p t as a readonly modifier.
/// \param  t The instance to wrap.
/// \tparam T The type to wrap in a modifier.
template <typename T, typename type_t = remove_ref_t<T>>
auto as_readonly(T& t) noexcept
  -> ModificationSpecifier<type_t&, Modifier::readonly> {
  return ModificationSpecifier<type_t&, Modifier::readonly>{t};
}

/// Creates a wrapped \p t as a writable modifier.
/// \param  t The instance to wrap.
/// \tparam T The type to wrap in a modifier.
template <typename T, typename type_t = remove_ref_t<T>>
auto as_writable(T& t) noexcept
  -> ModificationSpecifier<type_t&, Modifier::writable> {
  return ModificationSpecifier<type_t&, Modifier::writable>{t};
}

/// Creates a wrapped \p t as writable and which uses shared memory.
/// \param  t The instance to wrap.
/// \tparam T The type to wrap in a modifier.
template <typename T, typename type_t = remove_ref_t<T>>
auto as_writable_in_shared(T& t) noexcept
  -> ModificationSpecifier<type_t&, Modifier::writable_shared> {
  return ModificationSpecifier<type_t&, Modifier::writable_shared>{t};
}

/// Creates a wrapped \p t as readonly and which uses shared memory.
/// \param  t The instance to wrap.
/// \tparam T The type to wrap in a modifier.
template <typename T, typename type_t = remove_ref_t<T>>
auto as_readonly_in_shared(T& t) noexcept
  -> ModificationSpecifier<type_t&, Modifier::readonly_shared> {
  return ModificationSpecifier<type_t&, Modifier::readonly_shared>{t};
}

/// Creates a wrapped \p t as a shared modifier.
/// \param  t The instance to wrap.
/// \tparam T The type to wrap in a modifier.
template <typename T, typename type_t = remove_ref_t<T>>
auto in_shared(T& t) noexcept
  -> ModificationSpecifier<type_t&, Modifier::shared> {
  return ModificationSpecifier<type_t&, Modifier::shared>{t};
}

/// Returns a reference to objects wrapped by the modififer. This overload
/// simply forwards the type, and will be selected in overload resolution for
/// any type which is not a modifier.
/// \tparam T The type to unwrap.
template <typename T>
auto unwrap(T&& t) noexcept -> decltype(std::forward<T>(t)) {
  return std::forward<T>(t);
}

/// Returns a reference to the object wrapped by the modifier.
/// \tparam T The type wrapped by the modifier.
/// \tparam M The type of the modification.
template <typename T, Modifier M>
auto unwrap(ModificationSpecifier<T, M> t) noexcept
  -> remove_ref_t<decltype(t.wrapped)> {
  return t.wrapped;
}

} // namespace ripple

#endif // RIPPLE_GRAPH_MODIFIER_HPP

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

namespace ripple {

/**
 * Defines the type of the modifier.
 */
enum class Modifier : uint8_t {
  concurrent                 = 0,  //!< Concurrent modification.
  exclusive                  = 1,  //!< Exclusive modification.
  shared                     = 2,  //!< Uses shared memory.
  concurrent_shared          = 3,  //!< Concurrent and uses shared memory.
  exclusive_shared           = 4,  //!< Exclusive and uses shared memory.
  expander                   = 5,  //!< Expands into padding.
  concurrent_expander        = 6,  //!< Concurrent & expander
  exclusive_expander         = 7,  //!< Exclusive & expander
  shared_expander            = 8,  //!< Shared & expander.
  concurrent_shared_expander = 9,  //!< Concurrent & shared & expander.
  exclusive_shared_expander  = 10, //!< Exclusive & shared  & expander
};

using ExpType = int16_t;

/**
 * The ModificationSpecifier struct wraps a type T with some information for
 * how it is modified.
 *
 * \tparam T            The type which is wrapped in the modifier.
 * \tparam Modification The type of the modification.
 */
template <typename T, Modifier Modification = Modifier::concurrent>
struct ModificationSpecifier {
  T       wrapped;       //!< The type being wrapped.
  ExpType expansion = 0; //!< The amount to expand into the padding.
  ExpType overlap   = 0; //!< The amount of overlap.
};

/*==--- [traits] -----------------------------------------------------------==*/

/**
 * Modification traits for a type which is not a modifier.
 * \tparam T The type to get the traits for.
 */
template <typename T>
struct ModificationTraits {
  /** Defines the wrapped type for the modifier. */
  using Value = std::remove_reference_t<T>;

  // clang-format off
  /** Returns true if T is a modifier type. */
  static constexpr bool is_modifier             = false;
  /** Returns true if the modification access is concurrent. */
  static constexpr bool is_concurrent           = true;
  /** Returns true if the modification access is exclusive. */
  static constexpr bool is_exclusive            = !is_concurrent;
  /** Returns true if the modification requires shared memory. */
  static constexpr bool uses_shared             = false;
  /** Returns true if the modifier is exclusive or concurrent. */
  static constexpr bool exclusive_or_concurrent = false;
  /** Returns true if the modifier is an expander. */
  static constexpr bool is_expander             = false;
  // clang-format on
};

/**
 * Modification traits for a type which is a modifier.
 * \tparam T            The type wrapped by the modifier.
 * \tparam Modification The type of modification for the modifier.
 */
template <typename T, Modifier Modification>
struct ModificationTraits<ModificationSpecifier<T, Modification>> {
  /** Defines the wrapped type for the modifier. */
  using Value = std::remove_reference_t<T>;

  // clang-format off
  /** Returns true that this is a modifier type */
  static constexpr bool is_modifier = true;

  /** Returns true if the modification is concurrent. */
  static constexpr bool is_concurrent = 
    Modification == Modifier::concurrent          ||
    Modification == Modifier::concurrent_shared   ||
    Modification == Modifier::concurrent_expander ||
    Modification == Modifier::concurrent_shared_expander;

  /** Returns true if the modification is exclusive. */
  static constexpr bool is_exclusive = 
    Modification == Modifier::exclusive          ||
    Modification == Modifier::exclusive_shared   ||
    Modification == Modifier::exclusive_expander ||
    Modification == Modifier::exclusive_shared_expander;

  /** Returns true if the modification requires shared memory. */
  static constexpr bool uses_shared =
    Modification == Modifier::shared                    ||
    Modification == Modifier::exclusive_shared          ||
    Modification == Modifier::concurrent_shared         ||
    Modification == Modifier::shared_expander           ||
    Modification == Modifier::exclusive_shared_expander ||
    Modification == Modifier::concurrent_shared_expander;

  /** Returns true if this is an expander modifier. */
  static constexpr bool is_expander   = 
    Modification == Modifier::expander                  ||
    Modification == Modifier::shared_expander           ||
    Modification == Modifier::concurrent_expander       ||
    Modification == Modifier::exclusive_expander        ||
    Modification == Modifier::exclusive_shared_expander ||
    Modification == Modifier::concurrent_shared_expander;


  /** Returns true if the modification is exclusive or concurrent. */
  static constexpr bool exclusive_or_concurrent = is_exclusive || is_concurrent;

  // clang-format on
};

/**
 * Alias for the modification traits for T.
 * \tparam T The type to get the modification traits for.
 */
template <typename T>
using modification_traits_t = ModificationTraits<std::decay_t<T>>;

/**
 * Determines if the given type is a modifier.
 * \tparam T The type to determine if is a modifier.
 */
template <typename T>
static constexpr bool is_modifier_v = modification_traits_t<T>::is_modifier;

/**
 * Determines if the given type is an expander modifier.
 * \tparam T The type to determine if is an expander.
 */
template <typename T>
static constexpr bool is_expander_modifier_v =
  modification_traits_t<T>::is_expander;

/**
 * Returns true if any of the template parameters have either exclusive or
 * concurrent modifiers.
 * \tparam Modifiers The modifiers to determive if have a modifier.
 */
template <typename... Modifiers>
static constexpr bool has_modifier_v =
  (modification_traits_t<Modifiers>::exclusive_or_concurrent || ... || false);

/**
 * Defines a valid type if the template parameter is a modfier which specifies
 * shared memory use.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using shared_mod_enable_t =
  std::enable_if_t<modification_traits_t<T>::uses_shared, int>;

/**
 * Defines a valid type if the template parameter is not a modfier which
 * specifiesshared memory use.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_shared_mod_enable_t =
  std::enable_if_t<!modification_traits_t<T>::uses_shared, int>;

/**
 * Defines a valid type if it is a modifier.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using modifier_enable_t = std::enable_if_t<is_modifier_v<T>, int>;

/**
 * Defines a valid type if T is not a modifier.
 * \tparam T The type to base the enable on.
 */
template <typename T>
using non_modifier_enable_t = std::enable_if_t<!is_modifier_v<T>, int>;

/*==--- [methods] ----------------------------------------------------------==*/

/**
 * Parameters specifying the type of expansion.
 */
struct ExpansionParams {
  ExpType expansion = 0; //!< The amount of expansion into padding.
  ExpType overlap   = 0; //!< The amount of overlapping cells.
};

/**
 * Creates expansion parameters for the given expansion.
 * \param The amount of expansion on each side of each dimension.
 * \return The expansion paramters defining the expansion.
 */
inline auto with_expansion(ExpType expansion) noexcept -> ExpansionParams {
  return ExpansionParams{expansion, expansion};
}

/**
 * Creates expansion parameters for the given overlap.
 * \param The amount of overlap for each block of execution.
 * \return The expansion paramters defining the overlap.
 */
inline auto with_overlap(ExpType overlap) noexcept -> ExpansionParams {
  return ExpansionParams{overlap, overlap};
}

/**
 * Creates a wrapped type as a concurrent modifier which requies padding.
 * \param  t The instance to wrap.
 * \tparam T The type to wrap in a modifier.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto concurrent_padded_access(T& t) noexcept {
  return ModificationSpecifier<Type&, Modifier::concurrent>{t};
}

/**
 * Creates a wrapped type as a concurrent modifier which requies padding.
 * \param  t The instance to wrap.
 * \tparam T The type to wrap in a modifier.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto concurrent_padded_access(T& t, ExpansionParams params) noexcept {
  return ModificationSpecifier<Type&, Modifier::concurrent_expander>{
    t, params.expansion, params.overlap};
}

/**
 * Creates a wrapped type as an exclusive modifier which required padding.
 * \param  t The instance to wrap.
 * \tparam T The type to wrap in a modifier.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto exclusive_padded_access(T& t) noexcept {
  return ModificationSpecifier<Type&, Modifier::exclusive>{t};
}

/**
 * Creates a wrapped type as an exclusive modifier which required padding.
 * \param  t The instance to wrap.
 * \tparam T The type to wrap in a modifier.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto exclusive_padded_access(T& t, ExpansionParams params) noexcept {
  return ModificationSpecifier<Type&, Modifier::exclusive_expander>{
    t, params.expansion, params.overlap};
}

/**
 * Creates a wrapped type as a concurrent modififer which uses shared memory and
 * which requires padding.
 * \param  t         The instance to wrap.
 * \param  expantion Number of cells to expand the domain size by, reducing the
 *                   padding by this amount on each size of each dimension.
 * \tparam T         The type to wrap in a modifier.
 * \return The type wrapped in the modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto concurrent_padded_access_in_shared(T& t) noexcept {
  return ModificationSpecifier<Type&, Modifier::concurrent_shared>{t};
}

/**
 * Creates a wrapped type as a concurrent modififer which uses shared memory and
 * which requires padding.
 * \param  t         The instance to wrap.
 * \param  expantion Number of cells to expand the domain size by, reducing the
 *                   padding by this amount on each size of each dimension.
 * \tparam T         The type to wrap in a modifier.
 * \return The type wrapped in the modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto concurrent_padded_access_in_shared(T& t, ExpansionParams params) noexcept {
  return ModificationSpecifier<Type&, Modifier::concurrent_shared_expander>{
    t, params.expansion, params.overlap};
}

/**
 * Creates a wrapped type as an exclusive modifier which uses shared memory and
 * which requires padding.
 * \param  t         The instance to wrap.
 * \param  expantion Number of cells to expand the domain size by, reducing the
 *                   padding by this amount on each size of each dimension.
 * \tparam T         The type to wrap in a modifier.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto exclusive_padded_access_in_shared(T& t) noexcept {
  return ModificationSpecifier<Type&, Modifier::exclusive_shared>{t};
}

/**
 * Creates a wrapped type as an exclusive modifier which uses shared memory and
 * which requires padding.
 * \param  t         The instance to wrap.
 * \param  expantion Number of cells to expand the domain size by, reducing the
 *                   padding by this amount on each size of each dimension.
 * \tparam T         The type to wrap in a modifier.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto exclusive_padded_access_in_shared(T& t, ExpansionParams params) noexcept {
  return ModificationSpecifier<Type&, Modifier::exclusive_shared_expander>{
    t, params.expansion, params.overlap};
}

/**
 * Creates a wrapped type as a shared modifier.
 * \param  t The instance to wrap.
 * \tparam T The type to wrap in a modifier.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto in_shared(T& t) noexcept {
  return ModificationSpecifier<Type&, Modifier::shared>{t};
}

/**
 * Creates a wrapped type as a modifier which expands into the padding for
 * the data which is wrapper.
 * \param  t         The instance to wrap.
 * \param  expantion Number of cells to expand the domain size by, reducing the
 *                   padding by this amount on each size of each dimension.
 * \tparam T         The type of the instance.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto expanded(T& t, ExpType expansion) noexcept {
  return ModificationSpecifier<Type&, Modifier::expander>{
    t, expansion, expansion};
}

/**
 * Creates a wrapped type as a modifier which expands into the padding for
 * the data which is wrapper.
 * \param  t         The instance to wrap.
 * \param  expantion Number of cells to expand the domain size by, reducing the
 *                   padding by this amount on each size of each dimension.
 * \tparam T         The type of the instance.
 * \return The type wrapped in a modifier.
 */
template <typename T, typename Type = std::remove_reference_t<T>>
auto overlapped(T& t, ExpType overlap) noexcept {
  return ModificationSpecifier<Type&, Modifier::expander>{t, overlap, overlap};
}

/**
 * Gets a reference to objects wrapped by the modififer.
 *
 * \note This overload simply forwards the type, and will be selected in
 *       overload resolution for any type which is not a modifier.
 *
 * \tparam T The type to unwrap.
 * \return An rvalue reference to the argument.
 */
template <typename T>
auto unwrap_modifiers(T&& t) noexcept -> T&& {
  return static_cast<T&&>(t);
}

/**
 * Gets a reference to the object wrapped by the modifier.
 *
 * \note This overload is for modification specifiers, and acrually returns the
 *       reference to the wrapped type.
 *
 * \tparam T The type wrapped by the modifier.
 * \tparam M The type of the modification.
 * \return A reference to the wrapped instance.
 */
template <typename T, Modifier M>
auto unwrap_modifiers(ModificationSpecifier<T, M> t) noexcept
  -> std::remove_reference_t<decltype(t.wrapped)>& {
  return t.wrapped;
}

/**
 * Gets the expansion paramters if the type is a modifier.
 * \tparam T The type to unwrap.
 * \return The expansion parameters representing the expansion.
 */
template <typename T>
auto get_modifier_expansion_params(const T& t) noexcept -> ExpansionParams {
  if constexpr (is_modifier_v<T>) {
    return ExpansionParams{t.expansion, t.overlap};
  } else {
    return ExpansionParams{};
  }
}

} // namespace ripple

#endif // RIPPLE_GRAPH_MODIFIER_HPP

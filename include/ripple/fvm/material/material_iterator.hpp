//==--- ripple/fvm/material/material_iterator.hpp ---------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_iterator.hpp
/// \brief This file defines a type which can iterate over material data.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MATERIAL_MATERIAL_ITERATOR_HPP
#define RIPPLE_MATERIAL_MATERIAL_ITERATOR_HPP

#include <ripple/fvm/eos/eos_traits.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>

namespace ripple::fv {

/// The MaterialIterator class is essentially a wrapper class to wrap a state
/// and levelset iterator into a single interface, along with an equation of
/// state which is valid for all state data which is insine the levelset.
///
/// This should be created through the ``make_material_iterator`` function.
///
/// \tparam StateIterator    The type of the state iterator.
/// \tparam LevelsetIterator The type of the levelset iterator.
/// \tparam Eos              The equation of state for the material.
template <typename StateIterator, typename LevelsetIterator, typename Eos>
class MaterialIterator {
  /// Defines the type of the state iterator.
  using state_iter_t    = StateIterator;
  /// Defines the type of the levelset iterator.
  using levelset_iter_t = LevelsetIterator;
  /// Defines the type of the equation of state.
  using eos_t           = Eos;
  /// Defines the type of this iterator.
  using self_t          = MaterialIterator;

  state_iter_t    _state_iter;    //!< The state iterator for the material.
  levelset_iter_t _levelset_iter; //!< The levelset iterator for the material.
  eos_t           _eos;           //!< The equation of state for the material.

 public:
  /// Creates the material iterator, passing in a \p state and \p levelset
  /// iterator, as well as an \p eos equation of state.
  MaterialIterator(state_iter_t state, levelset_iter_t levelset, eos_t eos)
  : _state_iter{state}, _levelset_iter{levelset}, _eos{eos} {}

  //==--- [interface] ------------------------------------------------------==//
  
  /// Returns an iterator to the levelset for the material.
  ripple_host_device auto levelset() const -> levelset_iter_t {
    return _levelset_iter;
  }

  /// Returns an iterator to the state for the material.
  ripple_host_device auto state() const -> state_iter_t {
    return _state_iter;
  }

  //==--- [offsetting] -----------------------------------------------------==//

  /// Offsets the iterator by \p amount positions in the \p dim dimension,
  /// returning a new iterator offset to the location.
  /// \param  dim    The dimension to offset in
  /// \param  amount The amount to offset by.
  /// \tparam Dim    The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto offset(Dim&& dim, int amount = 1) const 
  -> self_t {
    return self_t{
      _state_iter.offset(std::forward<Dim>(dim), amount),
      _levelset_iter.offset(std::forward<Dim>(dim), amount),
      _eos
    };
  }

  /// Shifts the iterator by \p amount positions in the \p dim dimension.
  /// \param  dim    The dimension to offset in
  /// \param  amount The amount to offset by.
  /// \tparam Dim    The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto shift(Dim&& dim, int amount = 1)
  -> void {
    _state_iter.shift(std::forward<Dim>(dim), amount);
    _levelset_iter.shift(std::forward<Dim>(dim), amount);
  }
};

//==--- [functions] --------------------------------------------------------==//

/// Creates a material iterator with the \p state_iter iterator over state data,
/// and with the \p levelset_iter iterator over the levelset data, with the
/// given \p eos equation of state.
///
/// This will result in a compile time error if either of the iterators are not
/// iterators, or if they have different number of dimensions.
///
/// \param state_iter        An iterator over state data.
/// \param levelset_iter     An iterator over levelset data.
/// \param eos               The equation of state for the material iterator.
/// \tparam StateIterator    The type of the state iterator.
/// \param  LevelsetIterator The type of the levelset iterator.
/// \tparam Eos              The type of the equation of state implementation.
template <typename StateIterator, typename LevelsetIterator, typename Eos>
auto make_material_iterator(
  StateIterator&&    state_iter   ,
  LevelsetIterator&& levelset_iter,
  Eos&               eos
) -> MaterialIterator<
  std::decay_t<StateIterator>, std::decay_t<LevelsetIterator>, Eos
> {
  using mat_iter_t = MaterialIterator<
    std::decay_t<StateIterator>, std::decay_t<LevelsetIterator>, Eos
  >;
  using siter_traits_t = iterator_traits_t<StateIterator>;
  using liter_traits_t = iterator_traits_t<LevelsetIterator>;

  static_assert(
    is_iterator_v<StateIterator>, "Material iterator requires an iterator!"
  );
  static_assert(
    is_iterator_v<LevelsetIterator>, "Material iterator requires an iterator!"
  );

  constexpr auto dim_match = 
    iterator_traits_t<StateIterator>::dimensions ==
    iterator_traits_t<LevelsetIterator>::dimensions;
  static_assert(dim_match, "Iterators must have the same dimensionality!");

  return mat_iter_t{state_iter, levelset_iter, eos};
}

} // namespace ripple::fv

#endif // RIPPLE_MATERIAL_MATERIAL_ITERATOR_HPP


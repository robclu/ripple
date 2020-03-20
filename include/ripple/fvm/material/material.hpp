//==--- ripple/fvm/material/material.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material.hpp
/// \brief This file defines a type which represents a material.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MATERIAL_MATERIAL_HPP
#define RIPPLE_MATERIAL_MATERIAL_HPP

#include "material_iterator.hpp"
#include <ripple/fvm/eos/eos_traits.hpp>
#include <ripple/fvm/state/state_traits.hpp>
#include <ripple/core/container/grid.hpp>
#include <ripple/core/functional/pipeline.hpp>

namespace ripple::fv {

/// The Material type defines a type which has an equation of state for the
/// material, a state type for the data for the material, and a levelset which
/// defines the region to which the material is contained (i.e where the state
/// data is valid, and where it is not). The material class stores all data for
/// the material
/// \tparam State    The type of the state for the material.
/// \tparam Levelset The type of the levelset for the material.
/// \tparam Eos      The type of the equation of state for the material.
template <typename State, typename Levelset, typename Eos>
class Material {
 public:
  /// Defines the type of the equation of state.
  using eos_t           = Eos;
  /// Defines the type of the state for the material.
  using state_t         = State;
  /// Defines the type of the levelset for the material.
  using levelset_t      = Levelset;
  /// Defines the value type for the material.
  using value_t         = typename eos_traits_t<eos_t>::value_t;

 private:
  //==--- [constants] ------------------------------------------------------==//

  /// Defines the number of dimensions for the material.
  static constexpr auto dims = state_traits_t<State>::dimensions;
  /// Defines zero indices for dims dimensions;
  static constexpr size_t zero_indices[dims] = {0};

  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of the container used to store the data.
  using state_grid_t    = ripple::Grid<state_t, dims>;
  /// Defines the type of the levelset used to store the data.
  using levelset_grid_t = ripple::Grid<levelset_t, dims>;

 public:
  //==--- [constructor] ----------------------------------------------------==//
  
  /// Constructor which initializes __but does not allocate__ data for the
  /// states and the levelset. This constructor should be used when the size of
  /// the grid for the material __is not known__, and the material will be
  /// resized and its data reallocated at a later stage. 
  ///
  /// \param topo The system topology to use to allocate the data.
  /// \param eos  The equation of state for the material.
  Material(Topology& topo, eos_t eos)
  : _states(topo), _levelset(topo), _eos(eos) {}

  /// Constructor which initializes __and allocates__ the data for the states
  /// and levelset for the material to be the sizes defines by the \p sizes.
  /// This constructor is only enabled if the number of sizes is equal to the
  /// number of dimensions for the material (which is the number of dimensions
  /// for the state and levelset).
  ///
  /// \param  topo    The topology to use to initialize the data.
  /// \param  eos     The equation of state for the material.
  /// \param  sizes   The sizes of the dimensions for the material.
  /// \tparam Sizes   The types of the sizes for the material.
  template <
    typename... Sizes, all_arithmetic_size_enable_t<dims, Sizes...> =  0
  >
  Material(Topology& topo, eos_t eos, Sizes&&... sizes)
  : _states(topo, sizes...), _levelset(topo, sizes...), _eos(eos) {}

  //==--- [interface] ------------------------------------------------------==//
  
  /// Sets the padding for the material data to be \p width elements.
  /// \param width The number of elements for the padding for the material.
  auto set_padding(size_t width) -> void {
    _states.set_padding(width);

    // The levelset only ever requires a single padding cell.
    _levelset.set_padding(1);
  }
  
  /// Resizes the dimension of the material to \p size for the \p dim dimension.
  /// This does not allocate the data for the resized material, which needs to
  /// be done by calling ``reallocate()``.
  /// \param  dim  The dimension of the material to resize.
  /// \param  size The size of the dimension for the material.
  /// \tparam Dim  The type of the dimension specifier.
  template <typename Dim>
  auto resize_dim(Dim&& dim, size_t elements) -> void {
    _states.resize_dim(dim, elements);
    _levelset.resize_dim(dim, elements);
  }

  /// Reallocates all the data for the material. This is an expensive operation,
  /// and should not be performed in any critical performance region.
  auto reallocate() -> void {
    _states.reallocate();
    _levelset.reallocate();
  }

  /// Returns the size of the material in the \p dim dimension.
  /// \param  dim The dimension to get the size of.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  auto size(Dim&& dim) const -> size_t {
    return _states.size(std::forward<Dim>(dim));
  }

  /// Initializes the material using the \p levelset_init_func to initialize
  /// the levelset for the material, and the \p state_init_func to initialize
  /// the state data for the material, passing the eos for the material and 
  /// the \p state_initializers to the \p state_init_func function.
  /// 
  /// The \p levelset_init_func must have a signature of:
  ///
  /// ~~~{.cpp}
  /// // As a lambda:
  /// ripple_host_device auto (auto levelset) -> void
  ///
  /// // As a functor:
  /// template <typename T>
  /// ripple_host_device auto (T levelset) const -> void
  /// ~~~
  ///
  /// Where the type ``T`` or ``auto`` is a BlockIterator<Levelset, Dims>.
  ///
  /// The \p state_init_func must have a signature of:
  ///
  /// ~~~{.cpp}
  /// // As a lambda:
  /// ripple_host_device auto (auto state, auto eos, auto... inits) -> void
  ///
  /// // As a functor:
  /// template <typename T, typename EosImpl, typename... Inits>
  /// ripple_host_device auto 
  /// (T state, EosImpl&& eos, Inits&&... inits) const -> void
  /// ~~~
  ///
  /// Where the type ``T`` or ``auto`` is a BlockIterator<State, Dims>, and
  /// State is one of the implementations of the State<> interface, EosImpl
  /// implements the Eos<> interface, and Inits are StateInitializers.
  ///
  /// \param eos The equation of state for the material.
  /// \param  levelset_init_func The functor to initialize the levelset with.
  /// \param  state_init_func    The functor to initialize the state data with.
  /// \param  state_initializers The initializers for the state data.
  /// \tparam EosImpl            The implementation of the Eos interface.
  /// \tparam LevelsetInit       The type of the levelset initializer.
  /// \tparam StateInit          The type of the state initializer.
  /// \tparam StateInits         The type of the state initializers.
  template <typename LevelsetInit, typename StateInit, typename... StateInits>
  auto initialize(
    LevelsetInit&&  levelset_init_func,
    StateInit&&     state_init_func   ,
    StateInits&&... state_initializers
  ) -> void {
    _levelset.apply_pipeline(make_pipeline(levelset_init_func));
    _states.apply_pipeline_non_shared(
      make_pipeline(state_init_func),
      _eos                          , 
      std::forward<StateInits>(state_initializers)...
    );
  }

  //==--- [access] ---------------------------------------------------------==//
  
  /// Returns a reference to the equation of state.
  auto eos() -> eos_t& {
    return _eos;
  }

  /// Returns a const reference to the equation of state.
  auto eos() const -> const eos_t& {
    return _eos;
  }

  /// Returns a reference to the state data.
  auto states() -> state_grid_t& {
    return _states;
  }

  /// Returns a reference to the levelset grid data.
  auto levelset() -> levelset_grid_t& {
    return _levelset;
  }
  
  /// Returns an iterator over the state data for the material.
  auto state_iterator() -> typename state_grid_t::host_iter_t {
    return get_state_iterator(std::make_index_sequence<dims>());
  }

  /// Returns an iterator over the levelset data for the material.
  auto levelset_iterator() -> typename levelset_grid_t::host_iter_t {
    return get_levelset_iterator(std::make_index_sequence<dims>());
  }
  
  /// Returns a material iterator to the element at the \p indices, which is
  /// valid __only__ on the host.
  /// \param indices The indices of the element to get an iterator to.
  /// \param Indices The types of the indices.
  template <
    typename... Indices, all_arithmetic_size_enable_t<dims, Indices...> = 0
  >
  auto operator()(Indices&&... indices) {
    return make_material_iterator(
      _states(indices...), _levelset(indices...), _eos
    );
  }

 private:
  //==--- [members] --------------------------------------------------------==//
  state_grid_t    _states;    //!< State data for the material.
  levelset_grid_t _levelset;  //!< Levelset data for the material
  eos_t           _eos;       //!< The equation of state for the material.

  template <typename T, T... indices>
  auto get_state_iterator(std::integer_sequence<T, indices...>)
  -> typename state_grid_t::host_iter_t {
    return _states(zero_indices[indices]...);
  }

  template <typename T, T... indices>
  auto get_levelset_iterator(std::integer_sequence<T, indices...>)
  -> typename levelset_grid_t::host_iter_t {
    return _levelset(zero_indices[indices]...);
  }
};

} // namespace ripple::fv

#endif // RIPPLE_MATERIAL_MATERIAL_HPP

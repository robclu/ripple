//==--- ripple/fvm/solver/state_split_solver.hpp ----------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_split_solver.hpp
/// \brief This file defines a type which can advance the solution of iterated
///        state data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_SOLVER_STATE_SPLIT_SOLVER_HPP
#define RIPPLE_SOLVER_STATE_SPLIT_SOLVER_HPP

#include "solver.hpp"
#include <ripple/fvm/flux/flux.hpp>
#include <ripple/fvm/scheme/scheme.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple::fv {

/// The StateSplitSolver type implements the Solver interface to advance the
/// state data pointed to by an iterator.
/// \tparam FluxImpl   The implementation type of the Flux interface.
/// \tparam SchemeImpl The implementation type of the Scheme interface.
template <typename FluxImpl, typename SchemeImpl>
class StateSplitSolver : public Solver<StateSplitSolver<FluxImpl, SchemeImpl>> {
  /// Defines the type of the flux computer.
  using flux_t   = std::decay_t<FluxImpl>;
  /// Defines the type of the scheme for the solver.
  using scheme_t = std::decay_t<SchemeImpl>;

  /// Ensures that the equation of state implements the equation of state
  /// interface, the flux implements the Flux interface, and that the scheme
  /// implements the scheme interface.
  /// \tparam EosImpl The implementation type of the Eos interface.
  template <typename EosImpl>
  ripple_host_device constexpr auto ensure_interfaces_valid() const -> void {
    static_assert(is_eos_v<EosImpl>,
      "StateSplitSolver requires an Eos interface implementation."
    );
    static_assert(is_flux_v<flux_t>,
      "StateSplitSolver requires a Flux interface implementation."
    );
    static_assert(is_scheme_v<scheme_t>,
      "StateSplitSolver requires a Scheme interface implementation."
    );
  }

 public:
  /// Updates the \p in grid, storing the updated results in the \p out grid,
  /// and the \p loader to load the boundary data for the grids.
  ///
  /// This specific implementation is a split solver, and so solves over each of
  /// the dimensions, solving the dimension, and updating the solution. If the
  /// grid has two or more dimensions, the boundary data is loaded using the \p
  /// loader.
  ///
  /// \pre The \p in grid has boundaries which are loaded, otherwise this will
  ///      __not__ produce a correct result.
  ///
  /// \param  in      The input data to update.
  /// \param  out     The output data to store the results in.
  /// \param  eos     The equation of state for the data.
  /// \param  dt      The time discretization.
  /// \param  dh      The spatial discretization.
  /// \param  loader  The loader for the boundary data for the grid.
  /// \param  args    Additional arguments.
  /// \tparam T       The data type stored in the grid.
  /// \tparam Dims    The number of dimensions in the grid.
  /// \tparam U       The type of the discretization values.
  /// \tparam Loader  The type of the loader.
  /// \tparam EosImpl The equation of state implementation.
  /// \tparam Args    The types of additional arguments.
  template <
    typename    T      , 
    size_t      Dims   ,
    typename    U      ,
    typename    Loader ,
    typename    EosImpl,
    typename... Args
  >
  ripple_host_only auto update(
    Grid<T, Dims>& in    ,
    Grid<T, Dims>& out   ,
    U              dt    ,
    U              dh    ,
    Loader&&       loader, 
    EosImpl&       eos   ,
    Args&&...      args
  ) -> void {
    const auto pipeline = make_pipeline(*this);
    // Take references to the grids so that they can be swapped cheaply 
    // by changing the references, rather than actually swapping the grids.
    auto g_a = &in; auto g_b = &out;
    g_a->apply_pipeline(pipeline, *g_b, dt, dh, eos, dim_x);

    // Do the rest of the dimensions, swapping the grids each time.
    // For dim y, one pass has been done, so grids are swapped.
    // For dim z, two passes have been done, so grids are as the start.
    unrolled_for<Dims - 1>([&] (auto d) {
      constexpr auto dim = d == 0 ? dim_y : dim_z;
      g_a = dim == dim_y ? &out : &in;
      g_b = dim == dim_y ? &in  : &out;

      g_a->load_boundaries(loader);
      g_a->apply_pipeline(pipeline, *g_b, dt, dh, eos, dim);
    });

    // If there are two dimensions, we need to swap the input and output grids,
    // because the result is in the input grid.
    if constexpr (Dims == 2) {
      using std::swap;
      swap(in, out);
    }
  }

  /// Advances the state of the data which is iterated over by the \p it_in
  /// for the given time resolution \p dt, and spatial resolution \p dh, storing
  /// the solution in the \p it_out iterator.
  ///
  /// This will fail at compile time if either of the iterators do not implement
  /// the iterator interface, or if the equation of state does not implement the
  /// Eos interface.
  ///
  /// \param  it_in       An iterator over the input data to advance.
  /// \param  it_out      An iterator over the output data to advance.
  /// \param  dt          The time resolution for the update.
  /// \param  dh          The spatial resolution for the update.
  /// \param  eos         The equation of state for the data.
  /// \param  dim         The dimension to solve in.
  /// \param  args        Additional arguments.
  /// \tparam IteratorIn  The type of the input iterator.
  /// \tparam IteratorOut The type of the output iterator.
  /// \tparam T           The type of the resolution data.
  /// \tparam EosImpl     The implementation type of the Eos interface.
  /// \tparam Dim         The type of the dimension specifier.
  /// \tparam Args        The types of additional arguments.
  template <
    typename    IteratorIn ,
    typename    IteratorOut,
    typename    T          ,
    typename    EosImpl    ,
    typename    Dim        ,
    typename... Args
  >
  ripple_host_device auto operator()(
    IteratorIn&&  it_in , 
    IteratorOut&& it_out,
    T             dt    ,
    T             dh    ,
    EosImpl&&     eos   ,
    Dim&&         dim   ,
    Args&&...     args
  ) const -> void {
    this->template ensure_iterators_valid<IteratorIn, IteratorOut>();
    ensure_interfaces_valid<EosImpl>();

    constexpr auto scheme = scheme_t();
    *it_out = *it_in + 
      (dt / dh) * scheme.flux_delta(it_in, eos, flux_t(), dim, dt, dh);
  }
};

} // namespace ripple::fv

#endif // RIPPLE_SOLVER_STATE_SPLIT_SOLVER_HPP


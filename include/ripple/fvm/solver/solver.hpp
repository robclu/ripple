//==--- ripple/fvm/solver/solver.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  solver.hpp
/// \brief This file defines an interface for so;lver types.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_SOLVER_SOLVER_HPP
#define RIPPLE_SOLVER_SOLVER_HPP

#include <ripple/core/iterator/iterator_traits.hpp>
#include <ripple/core/utility/portability.hpp>

namespace ripple::fv {

/// The Solver type defines an interface for advancing the data for a problem.
/// \tparam Impl The implementation of the interface.
template <typename Impl>
class Solver {
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

  /// Returns a pointer to the implementation.
  ripple_host_device constexpr auto impl() -> impl_t* {
    return static_cast<impl_t*>(this);
  }

 protected:
  /// Ensures that the iterators are both iterators.
  /// \tparam IteratorIn  The type of the input iterator.
  /// \tparam IteratorOut The type of the output iterator. 
  template <typename IteratorIn, typename IteratorOut>
  ripple_host_device constexpr auto ensure_iterators_valid() const -> void {
    static_assert(is_iterator_v<IteratorIn>,
      "Sovler requires the input to be an iterator."
    );
    static_assert(is_iterator_v<IteratorOut>,
      "Sovler requires the output to be an iterator."
    );
  }

 public:
  /// Updates the \p in grid, storing the updated results in the \p out grid,
  /// using the \p loader to load the boundary data for the grids, if necessary.
  ///
  /// \param  in      The input data to update.
  /// \param  out     The output data to store the results in.
  /// \param  dt      The time discretization.
  /// \param  dh      The spatial discretization.
  /// \param  loader  The loader for the boundary data for the grid.
  /// \param  args    Additional arguments for implementations.
  /// \tparam T       The data type stored in the grid.
  /// \tparam Dims    The number of dimensions in the grid.
  /// \tparam U       The type of the discretization values.
  /// \tparam Loader  The type of the loader.
  /// \tparam Args    The types of additional arguments.
  template <
    typename T, size_t Dims, typename U, typename Loader, typename... Args
  >
  auto update(
    Grid<T, Dims>& in    ,
    Grid<T, Dims>& out   ,
    U              dt    ,
    U              dh    ,
    Loader&&       loader,
    Args&&...      args
  ) -> void {
    impl()->update(
      in, out, dt, dh, std::forward<Loader>(loader), std::forward<Args>(args)...
    );
  }


  /// Advances the state of the data which is iterated over by the \p i_in
  /// for the given time resolution \p dt, and spatial resolution \p dh, storing
  /// the solution in the \p it_out iterator.
  ///
  /// Specializations can use any additional \p args as required.
  ///
  /// This will fail at compile time if either of the iterators do not return
  /// true for `is_iterator_v`.
  ///
  /// \param  it_in       An iterator over the input data to advance.
  /// \param  it_out      An iterator over the output data to advance.
  /// \param  dt          The time resolution for the update.
  /// \param  dh          The spatial resolution for the update.
  /// \param  args        Additional arguments to use to advance the data.
  /// \tparam IteratorIn  The type of the input iterator.
  /// \tparam IteratorOut The type of the output iterator. 
  /// \tparam T           The type of the resolution data.
  /// \tparam Args        The type of additional arguments. 
  template <
    typename IteratorIn, typename IteratorOut, typename T, typename... Args
  >
  ripple_host_device auto operator()(
    IteratorIn&&  it_in ,
    IteratorOut&& it_out,
    T             dt    ,
    T             dh    ,
    Args&&...     args
  ) const -> void {
    ensure_iterators_valid<IteratorIn, IteratorOut>();
    impl()->operator()(
      std::forward<IteratorIn>(it_in)  ,
      std::forward<IteratorOut>(it_out),
      dt                               ,
      dh                               ,
      std::forward<Args>(args)...
    );
  }
};

//==--- [traits] -----------------------------------------------------------==//

/// Returns true if the type T implements the Solver interface, otherwise
/// returns false.
/// \tparam T The type to determine if implements the Solver interface.
template <typename T>
static constexpr auto is_solver_v =
  std::is_base_of_v<Solver<std::decay_t<T>>, std::decay_t<T>>;

} // namespace ripple::fv

#endif // RIPPLE_SOLVER_SPLIT_SOLVER_HPP

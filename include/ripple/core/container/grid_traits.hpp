//==--- ripple/core/container/grid_traits.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  grid_traits.hpp
/// \brief This file defines traits and forward declarations for grids.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_GRID_TRAITS_HPP
#define RIPPLE_CONTAINER_GRID_TRAITS_HPP

#include <ripple/core/utility/portability.hpp>

namespace ripple {

//==--- [forward declarations] ---------------------------------------------==//

/// The Grid class defines a class for a computational domain of any size,
/// comprised of blocks, which execute on either the host or the device.
///
/// \tparam T          The type of the data for the grid.
/// \tparam Dimensions The number of dimensions for the grid.
template <typename T, size_t Dimensions> class Grid;

//==--- [aliases] ----------------------------------------------------------==//

/// Alias for a 1-dimensional grid.
/// \tparam T The type of the data for the grid.
template <typename T>
using grid_1d_t = Grid<T, 1>;

/// Alias for a 2-dimensional grid.
/// \tparam T The type of the data for the grid.
template <typename T>
using grid_2d_t = Grid<T, 2>;

/// Alias for a 3-dimensional grid.
/// \tparam T The type of the data for the grid.
template <typename T>
using grid_3d_t = Grid<T, 3>;

} // namespace ripple

#endif // RIPPLE_CONTAINER_GRID_TRAITS_HPP


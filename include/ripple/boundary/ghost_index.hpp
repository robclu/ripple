//==--- ripple/boundary/ghost_index.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  ghost_index.hpp
/// \brief This file defines a simple class to represent an index of a ghost
///        cell.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_GHOST_INDEX_HPP
#define RIPPLE_BOUNDARY_GHOST_INDEX_HPP

#include <ripple/algorithm/unrolled_for.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/math/math.hpp>
#include <ripple/utility/dim.hpp>

namespace ripple {

//==--- [forward declaration] ----------------------------------------------==//

/// Forward declaration of the GhostIndex struct which stores the index of a
/// ghost cell in a number of dimensions.
/// \tparam Dimensions The number of dimensions for the ghost indices.
template <std::size_t Dimensions> struct GhostIndex;

//==--- [aliases] ----------------------------------------------------------==//

/// Defines a one dimensional ghost index type.
using ghost_index_1d_t = GhostIndex<1>;
/// Defines a two dimensional ghost index type.
using ghost_index_2d_t = GhostIndex<2>;
/// Defines a three dimensional ghost index type.
using ghost_index_3d_t = GhostIndex<3>;

//==--- [implementation] ---------------------------------------------------==//

/// Implementation of the GhostIndex struct to store the index of a ghost cell
/// in a number of dimensions.
///
/// For any dimension, the index of a ghost cell is the number of cells from
/// which it needs to be offset in the given dimension to reach the first non
/// ghost cell in the dimension. For example, in 2 dimensions, on the following
/// 2 x 2 domain, with 2 padding (ghost) cells, the ghost cell indices would be:
///
/// ~~~
///   ---------------------------------------------------------
///   | (2,2)  | (1,2)  |  (0,2) |  (0,2) |  (-1,2) |  (-2,2) |
///   ---------------------------------------------------------
///   | (2,1)  | (1,1)  |  (0,1) |  (0,1) |  (-1,1) |  (-2,1) |
///   ---------------------------------------------------------
///   | (2,0)  | (1,0)  |    x   |    x   |  (-1,0) |  (-2,0) |
///   ---------------------------------------------------------
///   | (2,0)  | (1,0)  |    x   |    x   |  (-1,0) |  (-2,0) |
///   ---------------------------------------------------------
///   | (2,-1) | (1,-1) | (0,-1) | (0,-1) | (-1,-1) | (-2,-1) |
///   ---------------------------------------------------------
///   | (2,-2) | (1,-2) | (0,-2) | (0,-2) | (-1,-2) | (-2,-2) |
///   ---------------------------------------------------------
/// ~~~
///
/// \tparam Dimensions The number of dimensions for the ghost indices.
template <std::size_t Dimensions>
struct GhostIndex {
  //==--- [traits] ---------------------------------------------------------==//
  
  /// Defines the type used for the indices.
  using value_t = int8_t;
  /// Defines the value of a void index.
  static constexpr auto void_value = value_t{0};

  //==--- [initialisation] -------------------------------------------------==//

 private:
  /// Initialises the indices from the \p it iterator in dimension \p dim with
  /// \p elements in the dimension. Returns true if the value was set to a valid
  /// value.
  /// \param  it         The iterator to use to set the ghost cell.
  /// \param  idx        The index in the dimension.
  /// \param  dim        The dimension to initialise.
  /// \param  size       The size of the dimension.
  /// \param  valid_cell Value to set to true if the cell is valid.
  /// \tparam Iterator   The type of the iterator.
  /// \tparam Dim        The dimension to initialise.
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto init_dim(
    Iterator&& it, std::size_t idx, Dim&& dim, size_t size, bool& valid_cell
  ) -> void {
    _values[dim] = std::min(idx, size - idx - 1);

    // Second condition for the case that there are more threads than the
    // iterator size in the dimension, and idx > it.size(), making the index
    // negative.
    if (_values[dim] < it.padding() && _values[dim] >= value_t{0}) {
      _values[dim] -= it.padding();
      const auto idx_i     = static_cast<int>(idx);
      const auto size_half = static_cast<int>(size) / 2;

      // Have to handle the case that the dimension is very small, i.e 2
      // elements, then idx_i - size_half = 0, rather than 1 which should be
      // the sign to use.
      _values[dim] *= idx_i == size_half ? 1 : math::sign(idx_i - size_half);
      valid_cell = true;
      return;
    } 
    set_as_void(dim);
  }

 public:
  
  /// Initialises the indices from the \p it iterator for the global domain,
  /// returning true if one of the indices for a dimension is valid, and hence
  /// that the index structure is valid (i.e that it defines a valid index for
  /// a ghost cell).
  /// \param it       The iterator to use to set the ghost cell.
  /// \param Iterator The type of the iterator.
  template <typename Iterator>
  ripple_host_device constexpr auto init_as_global(Iterator&& it) -> bool {
    bool loader_cell = false;
    unrolled_for<Dimensions>([&] (auto d) {
      constexpr auto dim = d;
      const auto     idx = global_idx(dim);
      init_dim(it, idx, dim, it.size(dim), loader_cell);
    });
    return loader_cell;
  }

  /// Initialises the indices from the \p it iterator for the block level
  /// domain, returning true if one of the indices for a dimension is valid, and
  /// hence that the index structure is valid (i.e that it defines a valid index
  /// for a ghost cell).
  /// \param it       The iterator to use to set the ghost cell.
  /// \param space    The space which defines the size of the block.
  /// \param Iterator The type of the iterator.
  template <typename Iterator ,typename Space>
  ripple_host_device constexpr auto init_as_block(
    Iterator&& it, Space&& space
  ) -> bool {
    bool loader_cell = false;
    unrolled_for<Dimensions>([&] (auto d) {
      constexpr auto dim = d;
      const auto     idx = thread_idx(dim);
      init_dim(it, idx, dim, space.size(dim), loader_cell);
    });
    return loader_cell;
  }

  //==--- [interface] ------------------------------------------------------==//
  
  /// Returns the number of dimensions for the ghost indices.
  ripple_host_device constexpr auto dimensions() const -> std::size_t {
    return Dimensions;
  }

  //==--- [access] ---------------------------------------------------------==//

  /// Returns the value of the index in the \p dim dimension.
  /// \param  dim The dimension to get the index for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto index(Dim&& dim) const -> value_t {
    return _values[dim];
  }

  /// Returns the a reference to the value of the index in the \p dim dimension.
  /// \param  dim The dimension to get the index for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto index(Dim&& dim) -> value_t& {
    return _values[dim];
  }

  /// Returns the absolute value of the index in the \p dim dimension.
  /// \param  dim The dimension to get the index for.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto abs_index(Dim&& dim) const -> value_t {
    return math::sign(_values[dim]) * _values[dim];
  }

  //==--- [utilities] ------------------------------------------------------==//
  
  /// Returns true if the index is a front index for dimension \p dim.
  /// \param  dim The dimension to determine if is a front index.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto is_front(Dim&& dim) const -> bool {
    return _values[dim] > value_t{0};
  }

  /// Returns true if the index is a back index for dimension \p dim.
  /// \param  dim The dimension to determine if is a front index.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto is_back(Dim&& dim) const -> bool {
    return _values[dim] < value_t{0};
  }

  /// Returns true if the index is neither back nor front, and therefore should
  /// nothing for the dimension.
  /// \param  dim The dimension to set as void.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto is_void(Dim&& dim) const -> bool {
    return _values[dim] == value_t{0};
  }

  /// Sets the index in dimension \p dim as a void index.
  /// \param  dim The dimension to set as a void index.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto set_as_void(Dim&& dim) -> bool {
    return _values[dim] = value_t{0};
  }

 private:
  int8_t _values[3] = {0}; //!< The values of the index.
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_GHOST_INDEX_HPP

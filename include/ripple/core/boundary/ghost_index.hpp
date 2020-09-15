//==--- ripple/core/boundary/ghost_index.hpp --------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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

#include <ripple/core/algorithm/unrolled_for.hpp>
#include <ripple/core/execution/thread_index.hpp>
#include <ripple/core/math/math.hpp>
#include <ripple/core/utility/dim.hpp>

namespace ripple {

/*==--- [forward declaration] ----------------------------------------------==*/

/**
 * Forward declaration of the GhostIndex struct which stores the index of a
 * ghost cell in a number of dimensions.
 * \tparam Dimensions The number of dimensions for the ghost indices.
 * \tparam MaxPadding The max padding value.
 */
template <size_t Dimensions>
struct GhostIndex;

/*==--- [aliases] ----------------------------------------------------------==*/

/** Defines a one dimensional ghost index type. */
using ghost_index_1d_t = GhostIndex<1>;
/** Defines a two dimensional ghost index type. */
using ghost_index_2d_t = GhostIndex<2>;
/** Defines a three dimensional ghost index type. */
using ghost_index_3d_t = GhostIndex<3>;

/*==--- [implementation] ---------------------------------------------------==*/

/**
 * Implementation of the GhostIndex struct to store the index of a ghost cell
 * in a number of dimensions.
 *
 * For any dimension, the index of a ghost cell is the number of cells from
 * which it needs to be offset in the given dimension to reach the first non
 * ghost cell in the dimension. For example, in 2 dimensions, on the following
 * 2 x 2 domain, with 2 padding (ghost) cells, the ghost cell indices would be:
 *
 *  ~~~
 *    ---------------------------------------------------------
 *    | (2,2)  | (1,2)  |  (0,2) |  (0,2) |  (-1,2) |  (-2,2) |
 *    ---------------------------------------------------------
 *    | (2,1)  | (1,1)  |  (0,1) |  (0,1) |  (-1,1) |  (-2,1) |
 *    ---------------------------------------------------------
 *    | (2,0)  | (1,0)  |    x   |    x   |  (-1,0) |  (-2,0) |
 *    ---------------------------------------------------------
 *    | (2,0)  | (1,0)  |    x   |    x   |  (-1,0) |  (-2,0) |
 *    ---------------------------------------------------------
 *    | (2,-1) | (1,-1) | (0,-1) | (0,-1) | (-1,-1) | (-2,-1) |
 *    ---------------------------------------------------------
 *    | (2,-2) | (1,-2) | (0,-2) | (0,-2) | (-1,-2) | (-2,-2) |
 *    ---------------------------------------------------------
 *  ~~~
 *
 * \tparam Dimensions The number of dimensions for the ghost indices.
 */
template <size_t Dimensions>
struct GhostIndex {
  /*==--- [traits] ---------------------------------------------------------==*/

  /** Defines the type used for the indices. */
  using value_t = int8_t;

  /** Defines the value of a void index. */
  static constexpr auto void_value = value_t{0};

  //==--- [initialisation] -------------------------------------------------==//

 private:
  /**
   * Initialises the indices from the \p it iterator in dimension \p dim with*
   * \p elements in the dimension.Returns true if the value was set to a valid
   * value.
   * \param it The iterator to use to set the ghost cell.
   * \param idx The index in thedimension.
   * \param dim The dimension to initialise.
   * \param size The size of the dimension.
   * \param valid_cell Value to set to true if the cell is valid.*
   * \tparam Iterator The type of the iterator.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto init_dim(
    Iterator&&  it,
    std::size_t idx,
    Dim&&       dim,
    size_t      size,
    bool&       valid_cell) noexcept -> void {
    // Set the value, and make sure that there is no overflow if the index
    // is too big, otherwise indices in the middle of a domain will think that
    // they need to load data!
    constexpr size_t max_padding = size_t{16};
    _values[dim]                 = static_cast<value_t>(
      std::min(max_padding, std::min(idx, size - idx - 1)));

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
  /**
   * Initialises the indices from the \p it iterator for the global domain,
   * returning true if one of the indices for a dimension is valid, and hence
   * that the index structure is valid (i.e that it defines a valid index for
   * a ghost cell).
   *
   * \param it       The iterator to use to set the ghost cell.
   * \param Iterator The type of the iterator.
   * \return true if one of the indicies for a dimension is valid.
   */
  template <typename Iterator>
  ripple_host_device constexpr auto
  init_as_global(Iterator&& it) noexcept -> bool {
    bool loader_cell = false;
    unrolled_for<Dimensions>([&](auto d) {
      constexpr auto dim = d;
      const auto     idx = it.global_idx(dim);
      init_dim(it, idx, dim, it.global_size(dim), loader_cell);
    });
    return loader_cell;
  }

  /**
   * Initialises the indices from the \p it iterator for the block level
   * domain, returning true if one of the indices for a dimension is valid, and
   * hence that the index structure is valid (i.e that it defines a valid index
   * for a ghost cell).
   *
   * \param it       The iterator to use to set the ghost cell.
   * \param space    The space which defines the size of the block.
   * \param Iterator The type of the iterator
   * \return true if one of the indicies for a dimension is valid.
   */
  template <typename Iterator, typename Space>
  ripple_host_device constexpr auto
  init_as_block(Iterator&& it, Space&& space) noexcept -> bool {
    bool loader_cell = false;
    unrolled_for<Dimensions>([&](auto d) {
      constexpr auto dim = d;
      const auto     idx = thread_idx(dim);
      init_dim(it, idx, dim, space.size(dim), loader_cell);
    });
    return loader_cell;
  }

  /*==--- [interface] -------------------------------------------------------=*/

  /**
   * Gets the number of dimensions for the ghost indices.
   * \return The number of dimensions of ghost indices.
   */
  ripple_host_device constexpr auto dimensions() const noexcept -> size_t {
    return Dimensions;
  }

  /*==--- [access] ---------------------------------------------------------==*/

  /**
   * Gets the value of the index in the \p dim dimension.
   * \param  dim The dimension to get the index for.
   * \tparam Dim The type of the dimension specifier.
   * \return The index in the dimension dim.
   */
  template <typename Dim>
  ripple_host_device constexpr auto index(Dim&& dim) const noexcept -> value_t {
    return _values[dim];
  }

  /**
   * Gets a reference to the value of the index in the \p dim dimension.
   * \param  dim The dimension to get the index for.
   * \tparam Dim The type of the dimension specifier.
   * \return An reference to the index for dimension dim.
   */
  template <typename Dim>
  ripple_host_device constexpr auto index(Dim&& dim) noexcept -> value_t& {
    return _values[dim];
  }

  /**
   * Gets the absolute value of the index in the \p dim dimension.
   * \param  dim The dimension to get the index for.
   * \tparam Dim The type of the dimension specifier.
   * \return The abs value of the index in dimension dim.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  abs_index(Dim&& dim) const noexcept -> value_t {
    return math::sign(_values[dim]) * _values[dim];
  }

  /*==--- [utilities] ------------------------------------------------------==*/

  /**
   * Determines if an index is at the from of the domain for dimension dim.
   * \param  dim The dimension to determine if is a front index.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the index is at the front of the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto is_front(Dim&& dim) const noexcept -> bool {
    return _values[dim] > value_t{0};
  }

  /**
   * Determines if the index is at the back of the domain for dimension dim.
   * \param  dim The dimension to determine if is a front index.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the index is at the back of the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto is_back(Dim&& dim) const noexcept -> bool {
    return _values[dim] < value_t{0};
  }

  /**
   *  Returns true if the index is neither back nor front, and therefore should
   *  nothing for the dimension.
   *
   * Determines if the index is void -- it is neither a front nor back index.
   * \param  dim The dimension to set as void.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the index is neither front nor back.
   */
  template <typename Dim>
  ripple_host_device constexpr auto is_void(Dim&& dim) const noexcept -> bool {
    return _values[dim] == void_value;
  }

  /**
   * Sets the index in dimension \p dim as a void index.
   * \param  dim The dimension to set as a void index.
   * \tparam Dim The type of the dimension specifier
   */
  template <typename Dim>
  ripple_host_device constexpr auto set_as_void(Dim&& dim) noexcept -> void {
    _values[dim] = void_value;
  }

 private:
  value_t _values[3] = {void_value}; //!< The values of the index.
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_GHOST_INDEX_HPP

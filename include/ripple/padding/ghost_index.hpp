/**=--- ripple/padding/ghost_index.hpp --------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  ghost_index.hpp
 * \brief This file defines a simple class to represent an index of a ghost
 *        cell.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_PADDING_GHOST_INDEX_HPP
#define RIPPLE_PADDING_GHOST_INDEX_HPP

#include <ripple/algorithm/unrolled_for.hpp>
#include <ripple/execution/thread_index.hpp>
#include <ripple/math/math.hpp>
#include <ripple/utility/dim.hpp>

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
using GhostIndex1d = GhostIndex<1>;
/** Defines a two dimensional ghost index type. */
using GhostIndex2d = GhostIndex<2>;
/** Defines a three dimensional ghost index type. */
using GhostIndex3d = GhostIndex<3>;

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
  /** Defines the type used for the indices. */
  using Value = int8_t;

  /** Defines the value of a void index. */
  static constexpr auto void_value = Value{0};

 private:
  /**
   * Initializes the indices from the it iterator in dimension dim with
   * elements in the dimension.
   *
   * \return true if the value was set to a valid value.
   * \param  it         The iterator to use to set the ghost cell.
   * \param  idx        The index in thedimension.
   * \param  dim        The dimension to initialise.
   * \param  size       The size of the dimension.
   * \param  valid_cell Value to set to true if the cell is valid.
   * \tparam Iterator   The type of the iterator.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto init_dim(
    Iterator&& it,
    size_t     idx,
    Dim&&      dim,
    size_t     size,
    bool&      valid_cell) noexcept -> void {
    /* Set the value, and make sure that there is no overflow if the index
     * is too big, otherwise indices in the middle of a domain will think that
     * they need to load data! */
    constexpr size_t max_padding = size_t{16};
    values_[dim] =
      static_cast<Value>(std::min(max_padding, std::min(idx, size - idx - 1)));

    /* Second condition for the case that there are more threads than the
     * iterator size in the dimension, and idx > it.size(), making the index
     * negative. */
    if (values_[dim] < it.padding() && values_[dim] >= Value{0}) {
      values_[dim] -= it.padding();
      const int idx_i     = static_cast<int>(idx);
      const int size_half = static_cast<int>(size) / 2;

      /* Have to handle the case that the dimension is very small, i.e 2
       * elements, then idx_i - size_half = 0, rather than 1 which should be
       * the sign to use. */
      values_[dim] *= idx_i == size_half ? 1 : math::sign(idx_i - size_half);
      valid_cell = true;
      return;
    }
    set_as_void(dim);
  }

 public:
  /**
   * Initialises the indices from the iterator for the global domain,
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
   * Initialises the indices from the iterator for the block level
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

  /**
   * Gets the number of dimensions for the ghost indices.
   * \return The number of dimensions of ghost indices.
   */
  ripple_host_device constexpr auto dimensions() const noexcept -> size_t {
    return Dimensions;
  }

  /*==--- [access] ---------------------------------------------------------==*/

  /**
   * Gets the value of the index in the dim dimension.
   * \param  dim The dimension to get the index for.
   * \tparam Dim The type of the dimension specifier.
   * \return The index in the dimension dim.
   */
  template <typename Dim>
  ripple_host_device constexpr auto index(Dim&& dim) const noexcept -> Value {
    return values_[dim];
  }

  /**
   * Gets a reference to the value of the index in the dim dimension.
   * \param  dim The dimension to get the index for.
   * \tparam Dim The type of the dimension specifier.
   * \return An reference to the index for dimension dim.
   */
  template <typename Dim>
  ripple_host_device constexpr auto index(Dim&& dim) noexcept -> Value& {
    return values_[dim];
  }

  /**
   * Gets the absolute value of the index in the dim dimension.
   * \param  dim The dimension to get the index for.
   * \tparam Dim The type of the dimension specifier.
   * \return The abs value of the index in dimension dim.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  abs_index(Dim&& dim) const noexcept -> Value {
    return math::sign(values_[dim]) * values_[dim];
  }

  /**
   * Determines if an index is at the from of the domain for dimension dim.
   * \param  dim The dimension to determine if is a front index.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the index is at the front of the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto is_front(Dim&& dim) const noexcept -> bool {
    return values_[dim] > Value{0};
  }

  /**
   * Determines if the index is at the back of the domain for dimension dim.
   * \param  dim The dimension to determine if is a front index.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the index is at the back of the dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto is_back(Dim&& dim) const noexcept -> bool {
    return values_[dim] < Value{0};
  }

  /**
   * Determines if the index is void -- it is neither a front nor back index.
   * \param  dim The dimension to set as void.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the index is neither front nor back.
   */
  template <typename Dim>
  ripple_host_device constexpr auto is_void(Dim&& dim) const noexcept -> bool {
    return values_[dim] == void_value;
  }

  /**
   * Sets the index in dimension dim as a void index.
   * \param  dim The dimension to set as a void index.
   * \tparam Dim The type of the dimension specifier
   */
  template <typename Dim>
  ripple_host_device constexpr auto set_as_void(Dim&& dim) noexcept -> void {
    values_[dim] = void_value;
  }

 private:
  /** The valud of the ghost indices. */
  Value values_[3] = {void_value, void_value, void_value};
};

} // namespace ripple

#endif // RIPPLE_PADDING_GHOST_INDEX_HPP

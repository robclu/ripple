/**=--- ripple/math/lerp.hpp ------------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  lerp.hpp
 * \brief This file defines linear interpolation functions.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_MATH_LERP_HPP
#define RIPPLE_MATH_LERP_HPP

#include "math.hpp"
#include <ripple/container/vec.hpp>
#include <ripple/iterator/iterator_traits.hpp>
#include <ripple/utility/range.hpp>

namespace ripple::math {

/*==--- [1d lerp] ----------------------------------------------------------==*/

/**
 * Interpolates the data around the iterator data using the given weights.
 *
 * Any weight for a dimension which is greater than 1 is interpreted to mean
 * shift the iterator over in the given dimension. For example, for the iterator
 * pointing to cell X, where the data is as follows:
 *
 * <pre class="diagram">
 *   *-----*-----*-----*-----*
 *   |  W  |  X  |  Y  |  Z  |
 *   *-----*-----*-----*-----*
 * </pre>
 *
 * as
 *   - $ w = {1.4}  : r = Y * (1.0 - 0.4) + Z * 0.4 $
 *   - $ w = {0.2}  : r = X * (1.0 - 0.2) + Y * 0.2 $
 *   - $ w = {-0.7} : r = X * {1.0 - 0.7} + W * 0.7 $
 *
 * where $w$ are the weights. This overload is enabled when the iterator is one
 * dimensional.
 *
 * This will fail at compile time if the iterator is not an iterator, or the
 * weights do not implement the Array interface.
 *
 * This overload is only enabled if the weight array is an array of size 1.
 *
 * This implementation uses the more numerically stable implementation
 * which is $ (1 - t) * v0 + t * v1 $, where $t$ is the weight, and $v0, v1$
 * are the two interpolation values, such that in the case that $w = 1$, the
 * returned value is `*it.offset(dimx(), 1)` and in the case that $w = 0$, the
 * the returned value is `*it`. Other implementations may not produce such a
 * result due to
 * [loss of significance](https://en.wikipedia.org/wiki/Loss_of_significance).
 *
 * \param  it       The iterator to use to interpolate data for.
 * \param  weights  The weights for the nodes in the interpolation.
 * \tparam Iterator The type of the iterator.
 * \tparam Weights  The type of the weights.
 * \return The underlying type of the iterator which is interpolated.
 */
template <
  typename Iterator,
  typename Weights,
  array_size_enable_t<Weights, 1> = 0>
ripple_host_device auto lerp(Iterator&& it, const Weights& weights) noexcept ->
  typename iterator_traits_t<Iterator>::CopyType {
  static_assert(
    is_iterator_v<Iterator>, "Linear interpolation requires an iterator.");
  using value_t = std::decay_t<decltype(weights[0])>;

  // Comptute values used for the interpolation:
  const auto sign    = math::sign(weights[dimx()]);
  const auto abs_w   = std::abs(weights[dimx()]);
  const auto abs_off = std::floor(abs_w);
  const auto off     = sign * abs_off;
  const auto factor  = abs_w - abs_off;

  return (*it.offset(dimx(), off)) * (value_t(1) - factor) +
         (*it.offset(dimx(), off + sign)) * factor;
}

/*==--- [2d lerp] ----------------------------------------------------------==*/

/**
 * Interpolates the data around the iterator using the given weights.
 *
 * Any weight for a dimension which is greater than 1 is interpreted to mean
 * shift the iterator over in the given dimension. For example, using the
 * following data:
 *
 * <pre class="diagram">
 *   *-----*-----*-----*-----*
 *   |  Y  |     |     |     |
 *   *-----*-----*-----*-----*
 *   |     |  A  |  B  |     |
 *   *-----*-----*-----*-----*
 *   |     |  C  |  D  |     |
 *   *-----*-----*-----*-----+*
 *   |  X  |     |     |     |
 *   *-----*-----*-----*-----+*
 * </pre>
 *
 * using weights of ${1.4, -1.7}$ from X will give:
 *
 * \begin{equation}
 *   close = C     * (1.0 - 0.4) + D   * 0.4
 *   far   = A     * (1.0 - 0.4) + B   * 0.4
 *   r     = close * (1.0 - 0.7) + far * 0.7
 * \end{equation}
 *
 * while using weights of ${1.4, 1.7}$ from Y will give:
 *
 * \begin{equation}
 *   close = A     * (1.0 - 0.4) + B   * 0.4
 *   far   = C     * (1.0 - 0.4) + D   * 0.4
 *   r     = close * (1.0 - 0.7) + far * 0.7
 * \end{equation}
 *
 * This will fail at compile time if the iterator is not an iterator, or the
 * weights do not implement the Array interface.
 *
 * This overload is only enabled if the size of the weight array is 2.
 *
 * This uses an implementation which does not have any loss of significance,
 * see the 1D implementation.
 *
 * \param  it       The iterator to use to interpolate data for.
 * \param  weights  The weights for the nodes in the interpolation.
 * \tparam Iterator The type of the iterator.
 * \tparam Weights  The type of the weights.
 * \return The underlying type of the iterator which is interpolated.
 */
template <
  typename Iterator,
  typename Weights,
  array_size_enable_t<Weights, 2> = 0>
ripple_host_device auto
lerp(const Iterator& it, const Weights& weights) noexcept ->
  typename iterator_traits_t<Iterator>::CopyType {
  static_assert(
    is_array_v<Weights>, "Linear interpolation requires a weight array.");
  static_assert(
    array_traits_t<Weights>::size,
    "Iterator dimensionality must match size of weight array.");

  using T = std::decay_t<decltype(weights[0])>;

  // Compute offset params:
  const T   absx    = std::abs(weights[dimx()]);
  const T   fl_absx = std::floor(absx);
  const int sign_x  = math::sign(weights[dimx()]);
  const int offx    = sign_x * fl_absx;
  const T   absy    = std::abs(weights[dimy()]);
  const T   fl_absy = std::floor(absy);
  const int sign_y  = math::sign(weights[dimy()]);
  const int offy    = sign_y * fl_absy;

  // Compute the factors:
  const T wx = absx - fl_absx;
  const T wy = absy - fl_absy;
  const T fx = T{1} - wx;
  const T fy = T{1} - wy;

  // clang-format off
  auto a = it.offset(dimx(), offx).offset(dimy(), offy);
  return 
      (*a)                         * (fx * fy)
    + (*a.offset(dimx(), sign_x))  * (wx * fy)
    + (*a.offset(dimy(), sign_y))  * (wy * fx)
    + (*a.offset(dimx(), sign_x)
         .offset(dimy(), sign_y))  * (wx * wy);

  // clang-format on
}

/*==--- [3d lerp] ----------------------------------------------------------==*/

/**
 * Interpolates the data around the iterator using the given weights.
 *
 * Any weight for a dimension which is greater than 1 is interpreted to mean
 * shift the iterator over in the given dimension. For example, using the
 * following data:
 *
 * <pre class="diagram">
 *     4       5
 *     *-------*
 *  0 /|    1 /|
 *   *-------* |
 *   | |7    | |6
 *   | *-----|-*
 *   |/      |/
 *   *-------*
 *  3        2
 * </pre>
 *
 * Given all positive weights, this will perform tri-linear interpolation,
 * first by doing a 2D linear interpolation in the $0->1->2->3$ plane to get
 * a result x, and another 2D linear interpolation in the $4->5->6->7$ plane,
 * to get a result y, and then a final 1D linear interpolation between x and
 * y will be done to get the result.
 *
 * This overload is enabled when the iterator is three dimensional.
 *
 * This will fail at compile time if the iterator is not an iterator, or the
 * weights do not implement the Array interface.
 *
 * This overload is only enabled if the size of the weight array is 3.
 *
 * This uses an implementation which does not have any loss of significance,
 * see the 1D implementation.
 *
 * \param  it       The iterator to use to interpolate data for.
 * \param  weights  The weights for the nodes in the interpolation.
 * \tparam Iterator The type of the iterator.
 * \tparam Weights  The type of the weights.
 * \return The underlying type of the iterator which is interpolated.
 */
template <
  typename Iterator,
  typename Weights,
  array_size_enable_t<Weights, 3> = 0>
ripple_host_device auto lerp(Iterator&& it, const Weights& weights) ->
  typename iterator_traits_t<Iterator>::CopyType {
  static_assert(
    is_array_v<Weights>, "Linear interpolation requires a weight array.");
  static_assert(
    array_traits_t<Weights>::size == 3,
    "Iterator dimensionality must match size of weight array.");
  using T = std::decay_t<decltype(weights[0])>;

  // Compute offset params:
  const T   absx    = std::abs(weights[dimx()]);
  const T   fl_absx = std::floor(absx);
  const int sign_x  = math::sign(weights[dimx()]);
  const int offx    = sign_x * fl_absx;
  const T   absy    = std::abs(weights[dimy()]);
  const T   fl_absy = std::floor(absy);
  const int sign_y  = math::sign(weights[dimy()]);
  const int offy    = sign_y * fl_absy;
  const T   absz    = std::abs(weights[dimz()]);
  const T   fl_absz = std::floor(absz);
  const int sign_z  = math::sign(weights[dimz()]);
  const int offz    = sign_z * fl_absz;

  // Compute the factors:
  const T wx = absx - fl_absx;
  const T wy = absy - fl_absy;
  const T wz = absz - fl_absz;
  const T fx = T{1} - wx;
  const T fy = T{1} - wy;
  const T fz = T{1} - wz;

  // Offset to the close (c) and far (f) cell in z plane:
  auto c = it.offset(dimx(), offx).offset(dimy(), offy).offset(dimz(), offz);
  auto f = c.offset(dimz(), sign_z);

  // clang-format off
  return 
    (*c)                                               * (fx * fy * fz) + 
    (*c.offset(dimx(), sign_x))                        * (wx * fy * fz) +
    (*c.offset(dimy(), sign_y))                        * (wy * fx * fz) +
    (*c.offset(dimx(), sign_x).offset(dimy(), sign_y)) * (wx * wy * fz) + 
    (*f)                                               * (fx * fy * wz) + 
    (*f.offset(dimx(), sign_x))                        * (wx * fy * wz) +
    (*f.offset(dimy(), sign_y))                        * (wy * fx * wz) +
    (*f.offset(dimx(), sign_x).offset(dimy(), sign_y)) * (wx * wy * wz);

  // clang-format on
}

} // namespace ripple::math

#endif // RIPPLE_MATH_LERP_HPP
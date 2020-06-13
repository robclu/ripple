//==--- ripple/core/math/lerp.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  lerp.hpp
/// \brief This file defines linear interpolation functions.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_MATH_LERP_HPP
#define RIPPLE_MATH_LERP_HPP

#include "math.hpp"
#include <ripple/core/container/vec.hpp>
#include <ripple/core/iterator/iterator_traits.hpp>
#include <ripple/core/utility/range.hpp>

namespace ripple::math {

//==--- [1d lerp] ----------------------------------------------------------==//

/// Interpolates the data around the \p it iterator data using the weights
/// defined by the \p weights. Any weight for a dimension which is greater than
/// 1 is interpreted to mean shift the iterator over in the given dimension. For
/// example, for the iterator pointing to cell X, where the data is as follows:
///
/// <pre class="diagram">
///   *-----*-----*-----*-----*
///   |  W  |  X  |  Y  |  Z  |
///   *-----*-----*-----*-----*
/// </pre>
///
/// as 
///   - $ w = {1.4}  : r = Y * (1.0 - 0.4) + Z * 0.4 $
///   - $ w = {0.2}  : r = X * (1.0 - 0.2) + Y * 0.2 $
///   - $ w = {-0.7} : r = X * {1.0 - 0.7} + W * 0.7 $
///
/// where $w$ are the weights. This overload is enables when the iterator is one
/// dimensional.
///
/// This will fail at compile time if the iterator is not an iterator, or the
/// weights do not implement the Array interface.
///
/// This overload is only enabled if the weight array is an array of size 1.
///
/// This implementation uses the more numerically stable implementation
/// which is $ (1 - t) * v0 + t * v1 $, where $t$ is the weight, and $v0, v1$
/// are the two interpolation values, such that in the case that $w = 1$, the
/// returned value is `*it.offset(dim_x, 1)` and in the case that $w = 0$, the
/// the returned value is `*it`. Other implementations may not produce such a 
/// result due to 
/// [loss of significance](https://en.wikipedia.org/wiki/Loss_of_significance).
///
/// \param  it       The iterator to use to interpolate data for.
/// \param  weights  The weights for the nodes in the interpolation.
/// \tparam Iterator The type of the iterator.
/// \tparam Weights  The type of the weights.
template <
  typename Iterator, typename Weights, array_size_enable_t<Weights, 1> = 0
>
ripple_host_device auto lerp(Iterator&& it, const Weights& weights) 
-> typename iterator_traits_t<Iterator>::copy_t {
  static_assert(
    is_iterator_v<Iterator>, "Linear interpolation requires an iterator."
  );
  using value_t = std::decay_t<decltype(weights[0])>;

  // Comptute values used for the interpolation:  
  const auto sign    = math::sign(weights[dim_x]);
  const auto abs_w   = std::abs(weights[dim_x]);
  const auto abs_off = std::floor(abs_w);
  const auto off     = sign * abs_off;
  const auto factor  = abs_w - abs_off;

  return 
    (*it.offset(dim_x, off))        * (value_t(1) - factor) +
    (*it.offset(dim_x, off + sign)) * factor;
}

//==--- [2d lerp] ----------------------------------------------------------==//

/// Interpolates the data around the \p it iterator using the weights
/// defined by the \p weights. Any weight for a dimension which is greater than
/// 1 is interpreted to mean shift the iterator over in the given dimension. For
/// example, using the following data:
///
/// <pre class="diagram">
///   *-----*-----*-----*-----*
///   |  Y  |     |     |     |
///   *-----*-----*-----*-----*
///   |     |  A  |  B  |     |
///   *-----*-----*-----*-----*
///   |     |  C  |  D  |     |
///   *-----*-----*-----*-----+*
///   |  X  |     |     |     |
///   *-----*-----*-----*-----+*
/// </pre>
///
/// using weights of ${1.4, -1.7}$ from X will give:
///
/// \begin{equation}
///   close = C     * (1.0 - 0.4) + D   * 0.4
///   far   = A     * (1.0 - 0.4) + B   * 0.4
///   r     = close * (1.0 - 0.7) + far * 0.7
/// \end{equation}
///
/// while using weights of ${1.4, 1.7}$ from Y will give:
///
/// \begin{equation}
///   close = A     * (1.0 - 0.4) + B   * 0.4
///   far   = C     * (1.0 - 0.4) + D   * 0.4
///   r     = close * (1.0 - 0.7) + far * 0.7
/// \end{equation}
///
/// This will fail at compile time if the iterator is not an iterator, or the
/// weights do not implement the Array interface.
///
/// This overload is only enabled if the size of the weight array is 2.
///
/// This uses an implementation which does not have any loss of significance,
/// see the 1D implementation.
///
/// \param  it       The iterator to use to interpolate data for.
/// \param  weights  The weights for the nodes in the interpolation.
/// \tparam Iterator The type of the iterator.
/// \tparam Weights  The type of the weights.
template <
  typename Iterator, typename Weights, array_size_enable_t<Weights, 2> = 0
>
ripple_host_device auto lerp(Iterator&& it, const Weights& weights)
-> typename iterator_traits_t<Iterator>::copy_t {
  static_assert(
    is_array_v<Weights>,"Linear interpolation requires a weight array."
  );
  constexpr auto elems = array_traits_t<Weights>::size;
  static_assert(
    elems == 2, "Iterator dimensionality must match size of weight array."
  );  
  //using value_t = std::decay_t<decltype(weights[0])>;
  using value_t = double;

  // Compute offset params:
  const auto sign_x   = math::sign(value_t{weights[dim_x]});
  const auto absx     = std::abs(value_t{weights[dim_x]});
  const auto fl_absx  = std::floor(absx);
  const auto offx     = sign_x * fl_absx;
  const auto sign_y   = math::sign(value_t{weights[dim_y]});
  const auto absy     = std::abs(value_t{weights[dim_y]});
  const auto fl_absy  = std::floor(absy);
  const auto offy     = sign_y * fl_absy;

  // Compute the factors:
  const auto wx = absx - fl_absx;
  const auto wy = absy - fl_absy;
  const auto fx = value_t{1} - wx;
  const auto fy = value_t{1} - wy;

  auto a = it.offset(dim_x, offx).offset(dim_y, offy);

  return (*a)                     * (fx * fy)
    + (*a.offset(dim_x, sign_x))  * (wx * fy)
    + (*a.offset(dim_y, sign_y))  * (wy * fx)
    + (*a.offset(dim_x, sign_x)
         .offset(dim_y, sign_y))  * (wx * wy);
}

//==--- [3d lerp] ----------------------------------------------------------==//

/// Interpolates the data around the \p it iterator using the weights
/// defined by the \p weights. Any weight for a dimension which is greater than
/// 1 is interpreted to mean shift the iterator over in the given dimension. For
/// example, using the following data:
///
/// <pre class="diagram">
///     4       5
///     *-------*
///  0 /|    1 /|
///   *-------* |
///   | |7    | |6
///   | *-----|-*
///   |/      |/
///   *-------*
///  3        2
/// </pre>
///
/// Given all positive weights, this will perform tri-linear interpolation, 
/// first by doing a 2D linear interpolation in the $0->1->2->3$ plane to get
/// a result x, and another 2D linear interpolation in the $4->5->6->7$ plane,
/// to get a result y, and then a final 1D linear interpolation between x and
/// y will be done to get the result. 
///
/// This overload is enabled when the iterator is two dimensional.
///
/// This will fail at compile time if the iterator is not an iterator, or the
/// weights do not implement the Array interface.
///
/// This overload is only enabled if the size of the weight array is 3.
///
/// This uses an implementation which does not have any loss of significance,
/// see the 1D implementation.
///
/// \param  it       The iterator to use to interpolate data for.
/// \param  weights  The weights for the nodes in the interpolation.
/// \tparam Iterator The type of the iterator.
/// \tparam Weights  The type of the weights.
template <
  typename Iterator, typename Weights, array_size_enable_t<Weights, 3> = 0
>
ripple_host_device auto lerp(Iterator&& it, const Weights& weights)
-> typename iterator_traits_t<Iterator>::copy_t {
  static_assert(
    is_array_v<Weights>, "Linear interpolation requires a weight array."
  );
  constexpr auto elems = array_traits_t<Weights>::size;
  static_assert(
    elems == 3, "Iterator dimensionality must match size of weight array."
  );  
  using value_t = std::decay_t<decltype(weights[0])>;

  // Compute offset params:
  const auto sign_x   = math::sign(weights[dim_x]);
  const auto absx     = std::abs(weights[dim_x]);
  const auto fl_absx  = std::floor(absx);
  const auto offx     = sign_x * fl_absx;
  const auto sign_y   = math::sign(weights[dim_y]);
  const auto absy     = std::abs(weights[dim_y]);
  const auto fl_absy  = std::floor(absy);
  const auto offy     = sign_y * fl_absy;
  const auto sign_z   = math::sign(weights[dim_z]);
  const auto absz     = std::abs(weights[dim_z]);
  const auto fl_absz  = std::floor(absz);
  const auto offz     = sign_z * fl_absz;

  // Compute the factors:
  const auto wx = absx - fl_absx;
  const auto wy = absy - fl_absy;
  const auto wz = absz - fl_absz;
  const auto fx = value_t{1} - wx;
  const auto fy = value_t{1} - wy;
  const auto fz = value_t{1} - wz;

  // Offset to the close (c) and far (f) cell in z plane:
  auto c = it.offset(dim_x, offx).offset(dim_y, offy).offset(dim_z, offz);
  auto f = c.offset(dim_z, sign_z);

  return 
    // x-y plane closest in z direction:
      (*c)                       * (fx * fy * fz)
    + (*c.offset(dim_x, sign_x)) * (wx * fy * fz)
    + (*c.offset(dim_y, sign_y)) * (wy * fx * fz)
    + (*c.offset(dim_x, sign_x)
         .offset(dim_y, sign_y)) * (wx * wy * fz)
    // x-y plane furthest in z direction:
    + (*f)                       * (fx * fy * wz)
    + (*f.offset(dim_x, sign_x)) * (wx * fy * wz)
    + (*f.offset(dim_y, sign_y)) * (wy * fx * wz)
    + (*f.offset(dim_x, sign_x)
         .offset(dim_y, sign_y)) * (wx * wy * wz);
}

} // namespace ripple::math

#endif // RIPPLE_MATH_LERP_HPP
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
///   +-----+-----+-----+-----+
///   |  W  |  X  |  Y  |  Z  |
///   +-----+-----+-----+-----+
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
/// weights do not implement the Array interface, and if the number of elements
/// in the array do not have the same dimensionality as the iterator.
///
/// This implementation uses the more numerically stable implementation
/// which is $ (1 - t) * v0 + t * v1 $, where $t$ is the weight, and $v0, v1$
/// are the two interpolation values, such that in the case that $w = 1$, the
/// returned value is `*it.offset(dim_x, 1)` and in the case that $w = 0$, the
/// the returned value is `*it`. Other implementations may not produce such a 
/// result due to 
/// [loss of significance](https://en.wikipedia.org/wiki/Loss_of_significance).
///
/// \pre This assumes that the \p it iterator is offset to the appropriate cell
///      from which the interpolation will be performed.
///
/// \param  it       The iterator to use to interpolate data for.
/// \param  weights  The weights for the nodes in the interpolation.
/// \tparam Iterator The type of the iterator.
/// \tparam Weights  The type of the weights.
template <typename Iterator, typename Weights, it_1d_enable_t<Iterator> = 0>
ripple_host_device auto lerp(Iterator&& it, const Weights& weights) 
-> typename iterator_traits_t<Iterator>::copy_t {
  static_assert(
    is_array_v<Weights>,"Linear interpolation requires a weight array."
  );
  constexpr auto dims  = iterator_traits_t<Iterator>::dimensions;
  constexpr auto elems = array_traits_t<Weights>::size;
  static_assert(
    dims == elems, "Iterator dimensionality must match size of weight array."
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

} // namespace ripple::math

#endif // RIPPLE_MATH_LERP_HPP

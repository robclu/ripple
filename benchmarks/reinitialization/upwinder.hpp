//==--- ripple/benchmarks/reinitialization/upwinder.hpp ---- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  upwinder.hpp
/// \brief This file defines an implementation of upwinding.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BENCHMARK_REINITIALIZATION_UPWINDER_HPP
#define RIPPLE_BENCHMARK_REINITIALIZATION_UPWINDER_HPP

#include "levelset_element.hpp"

/**
 * Forward declaration of upwinding class.
 */
template <size_t Dims>
struct Upwinder;

/**
 * Specialization of the upwinder for the 1D case.
 */
template <>
struct Upwinder<1> {
  /**
   * Invokes the upwinder on the iterator, when the solver is for a single
   * dimension.
   * \param  it The iterator to compute the upwinding from.
   * \param  dh The resolution of the domain.
   * \param  f  The speed function for the eikonal equation.
   * \tparam It The type of the iterator.
   * \tparam T  The type of the resolution.
   * \tparam F  The type of the speed function.
   */
  template <typename It, typename T, typename F>
  ripple_all auto operator()(It&& it, T dh, F&& f) const noexcept -> T {
    using namespace ripple;
    return std::min(
             it.offset(dimx(), -1)->value(), it.offset(dimx(), 1)->value()) +
           (f + dh);
  }
};

/**
 * Specialization of the upwinder for the 2D case.
 */
template <>
struct Upwinder<2> {
  /**
   * Invokes the upwinder on the iterator, when the solver is for two
   * dimensions.
   * \param  it The iterator to compute the upwinding from.
   * \param  dh The resolution of the domain.
   * \param  f  The speed function for the eikonal equation.
   * \tparam It The type of the iterator.
   * \tparam T  The type of the resolution.
   * \tparam F  The type of the speed function.
   */
  template <typename It, typename T, typename F>
  ripple_all auto operator()(It&& it, T dh, F&& f) const noexcept -> T {
    using namespace ripple;
    const T a =
      std::min(it.offset(dimx(), -1)->value(), it.offset(dimx(), 1)->value());
    const T b =
      std::min(it.offset(dimy(), -1)->value(), it.offset(dimy(), 1)->value());

    const T fh  = f * dh;
    const T amb = a - b;
    return std::abs(amb) >= fh
             ? std::min(a, b) + fh
             : T(0.5) * (a + b + std::sqrt(T(2) * fh * fh - (amb * amb)));
  }
};

/**
 * Specialization fo the upwinder for the 3D case.
 */
template <>
struct Upwinder<3> {
  /**
   * Invokes the upwinder on the iterator, when the solver is for three
   * dimensions.
   * \param  it The iterator to compute the upwinding from.
   * \param  dh The resolution of the domain.
   * \param  f  The speed function for the eikonal equation.
   * \tparam It The type of the iterator.
   * \tparam T  The type of the resolution.
   * \tparam F  The type of the speed function.
   */
  template <typename It, typename T, typename F>
  ripple_all auto operator()(It&& it, T dh, F&& f) const noexcept -> T {
    using namespace ripple;
    const T aa =
      std::min(it.offset(dimx(), -1)->value(), it.offset(dimx(), 1)->value());
    const T bb =
      std::min(it.offset(dimy(), -1)->value(), it.offset(dimy(), 1)->value());
    const T cc =
      std::min(it.offset(dimz(), -1)->value(), it.offset(dimz(), 1)->value());

    const T a = std::min(aa, std::min(bb, cc));
    const T b = std::max(std::min(aa, bb), std::min(std::max(aa, bb), cc));
    const T c = std::max(aa, std::max(bb, cc));

    const T fh = f * dh;
    T       r  = a + fh;

    if (r <= b) {
      return r;
    }

    r = a - b;
    r = T(0.5) * (a + b + std::sqrt(T(2) * fh * fh - (r * r)));
    if (r <= c) {
      return r;
    }

    r              = a + b + c;
    constexpr T fc = T(1) / T(6);
    return fc * (T(2) * r +
                 std::sqrt(
                   T(4) * r * r - T(12) * (a * a + b * b + c * c - fh * fh)));
  }
};

#endif // RIPPLE_BENCHMARK_REINITIALIZATION_UPWINDER_HPP
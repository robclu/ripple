//==--- ripple/fvm/eos/eos_traits.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  eos_traits.hpp
/// \brief This file defines traits and forward declarations for equation of 
///        state related functionality.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_EOS_EOS_TRAITS_HPP
#define RIPPLE_EOS_EOS_TRAITS_HPP

#include <ripple/core/utility/portability.hpp>

namespace ripple::fv {

//==--- [forward declarations] ---------------------------------------------==//

/// The IdealGas type implements the Eos interface to define and equation of
/// state which represents an ideal gas.
/// \tparam T The type of the data used for computation.
template <typename T> struct IdealGas;


/// The EosTraits type defines traits for an equation of state.
/// \tparam EqnOfState The equation of state to get the traits for.
template <typename EqnOfState> struct EosTraits;

//==--- [specializations]  -------------------------------------------------==//

/// Specialization of the equation of state traits for an ideal gas.
/// \tparam T The type of the data used by the ideal gas.
template <typename T>
struct EosTraits<IdealGas<T>> {
  /// Defines the data type used by the equation of state.
  using value_t = std::decay_t<T>;
};

} // namespace ripple::fv

#endif // RIPPLE_EOS_EOS_TRAITS_HPP


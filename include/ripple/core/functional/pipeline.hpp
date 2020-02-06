//==--- ripple/core/functional/pipeline.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  pipeline.hpp
/// \brief This file implements functionality for a pipeline, which is a chain
///        of operations.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FUNCTIONAL_PIPELINE_HPP
#define RIPPLE_FUNCTIONAL_PIPELINE_HPP

#include "functional_traits.hpp"
#include "invocable.hpp"
#include <ripple/core/container/tuple.hpp>

namespace ripple {

/// The Pipeline class stores a chain of invocable objects, where each
/// strage in the pipeline defined by an invocable object.
///
/// \tparam Ts The types of the invocable objects.
template <typename... Ts>
class Pipeline {
  /// Defines the type of the container for the pipeline.
  using stage_container_t = Tuple<Ts...>;

 public:
  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines the number of stages in the pipeline.
  static constexpr auto num_stages = size_t{sizeof...(Ts)};
  
  //==--- [construction] ---------------------------------------------------==//
  
  /// Constructor to create the pipeline with the \p ops operations, by copying
  /// the operations into the pipeline.
  /// \tparam stages The stages to copy into the pipeline.
  ripple_host_device Pipeline(const Ts&... stages) noexcept 
  : _stages{stages...} {
    ensure_invocable();
  }
  
  /// Constructor to create the pipeline with the \p stages, by moving the
  /// stages into the pipeline.
  /// \tparam stages The stages to move into the pipeline.
  ripple_host_device Pipeline(Ts&&... stages) noexcept 
  : _stages{std::move(stages)...} {
    ensure_invocable();
  }

  //==--- [interface] ------------------------------------------------------==//
  
  /// Returns a reference to the Ith stage of the pipeline.
  /// \tparam I The index of the stage to get.
  template <size_t I>
  ripple_host_device auto get_stage() noexcept -> nth_element_t<I, Ts...>& {
    return get<I>(_stages);
  }

  /// Returns a const reference to the Ith stage in the pipeline.
  /// \tparam I The index of the stage to get.
  template <size_t I>
  ripple_host_device auto get_stage() const noexcept 
  -> const nth_element_t<I, Ts...>& {
    return get<I>(_stages);
  }

 private:
  stage_container_t _stages;  //!< Stages in the pipeline.

  //==--- [methods] --------------------------------------------------------==//

  /// Checks that all the operation types are invoccable.
  ripple_host_device constexpr auto ensure_invocable() const noexcept -> void {
    constexpr auto all_invocable = 
      std::conjunction_v<detail::IsInvocable<Ts>...>;
    static_assert(all_invocable, "Not all operation types are invocable!");
  }
}; 

//==--- [functions] --------------------------------------------------------==//

/// Creates a pipeline with \p stages stages.
/// \param  stages The stages for the pipeline.
/// \tparam Stages The type of the stages for the pipeline.
template <typename... Stages>
ripple_host_device auto make_pipeline(Stages&&... stages) 
-> Pipeline<make_invocable_t<Stages>...> {
  return Pipeline<make_invocable_t<Stages>...>{
    make_invocable_t<Stages>{stages}...
  };
}

} // namespace ripple

#endif // RIPPLE_FUNCTIONAL_PIPILINE_HPP


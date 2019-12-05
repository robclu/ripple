//==--- ripple/boundary/inernal_loader.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  internal_loader.hpp
/// \brief This file defines an interface for inernal loading of data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_INTERNAL_LOADER_HPP
#define RIPPLE_BOUNDARY_INTERNAL_LOADER_HPP

#include <ripple/utility/portability.hpp>

namespace ripple {

/// The InternalLoader class defines an interface for internally loading data
/// for an iterator from another iterator.
/// \tparam Impl The implementation of the loading interface.
template <typename Impl>
class InternalLoader {
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

 public:
  /// Loads the data in the \p it_to iterator using the data from the \p it_from
  /// iterator.
  /// \param  it_from      An iterator to cell to load from.
  /// \param  it_to        An iterator to cell to load into.
  /// \param  args         Additional arguments for the loading.
  /// \tparam IteratorFrom The type of the from iterator.
  /// \tparam IteratorTo   The type of the to iterator.
  /// \tparam Args         The types of the additional arguments.
  template <typename IteratorFrom, typename IteratorTo, typename... Args>
  ripple_host_device constexpr auto load(
    IteratorFrom&& it_from, IteratorTo&& it_to, Args&&... args
  ) const -> void {
    impl()->load(
      std::forward<IteratorFrom>(it_from),
      std::forward<IteratorTo>(it_to)    ,
      std::forward<Args>(args)...
    );
  }
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_INTERNAL_LOADER_HPP

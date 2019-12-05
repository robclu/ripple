//==--- ripple/boundary/boundary_loader.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  boundary_loader.hpp
/// \brief This file defines an interface for boundary loading.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_BOUNDARY_LOADER_HPP
#define RIPPLE_BOUNDARY_BOUNDARY_LOADER_HPP

#include <ripple/utility/portability.hpp>

namespace ripple {

/// The BoundaryLoader class defines an interface for boundary loading.
/// \tparam Impl The implementation of the loading interface.
template <typename Impl>
class BoundaryLoader {
  /// Defines the type of the implementation.
  using impl_t = std::decay_t<Impl>;

  /// Returns a const pointer to the implementation.
  ripple_host_device constexpr auto impl() const -> const impl_t* {
    return static_cast<const impl_t*>(this);
  }

 public:
  /// Loads the front boundary in the \p dim dimension, using the \p value of
  /// the \p index in the dimension.
  /// \param  it       An iterator to the boundary cell to load.
  /// \param  index    The index of the boundary cell in the dimension.
  /// \param  dim      The dimension to load the boundary in.
  /// \param  args     Additional arguments for the loading.
  /// \tparam Iterator The type of the iterator.
  /// \tparam Dim      The type of the dimension specifier.
  /// \tparam Args     The types of the additional arguments.
  template <typename Iterator, typename Dim, typename... Args>
  ripple_host_device constexpr auto load_front(
    Iterator&& it, int index, Dim&& dim, Args&&... args
  ) const -> void {
    impl()->load_front(
      std::forward<Iterator>(it),
      index                     ,
      std::forward<Dim>(dim)    ,
      std::forward<Args>(args)...
    );
  }

  /// Loads the back boundary in the \p dim dimension, using the \p value of
  /// the \p index in the dimension.
  /// \param  it       An iterator to the boundary cell to load.
  /// \param  index    The index of the boundary cell in the dimension.
  /// \param  dim      The dimension to load the boundary in.
  /// \param  args     Additional arguments for the loading.
  /// \tparam Iterator The type of the iterator.
  /// \tparam Dim      The type of the dimension specifier.
  /// \tparam Args     The types of the additional arguments.
  template <typename Iterator, typename Dim, typename... Args>
  ripple_host_device constexpr auto load_back(
    Iterator&& it, int index, Dim&& dim, Args&&... args
  ) const -> void {
    impl()->load_back(
      std::forward<Iterator>(it),
      index                     ,
      std::forward<Dim>(dim)    ,
      std::forward<Args>(args)...
    );
  }
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_BOUNDARY_LOADER_HPP

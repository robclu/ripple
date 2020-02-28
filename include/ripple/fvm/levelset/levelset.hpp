//==--- ripple/fvm/levelset/levelset.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset.hpp
/// \brief This file defines a class for a levelset.
///
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_LEVELSET_LEVELSET_HPP
#define RIPPLE_LEVELSET_LEVELSET_HPP

#include "levelset_traits.hpp"
#include <ripple/core/container/array.hpp>
#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/storage_traits.hpp>

namespace ripple::fv {

/// The Levelset type defines a class which stores scalar data in an N
/// dimensional space to represent the distance from an interface.
///
/// This class is essentially a handle for a levelset element which can be used
/// with a Block or a Grid to allocate the required levelset elements for an
/// entire space. It therefore does not own the storage.
///
/// \tparam T      The data type for the levelset data.
/// \tparam Dims   The number of dimensions for the levelset.
/// \tparam Layout The layout type for the data.
template <typename T, typename Dims, typename Layout>
class Levelset : public Array<Levelset<T, Dims, Layout>> {
  //==--- [constants] ------------------------------------------------------==//
  
  /// Defines the number of dimensions for the state.
  static constexpr auto dims           = size_t{Dims::value};
  /// Defines the total number of elements in the state.
  static constexpr auto elements       = 1;
  /// Defines the boundary value of the levelset.
  static constexpr auto boundary_value = T{0};

  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of this state.
  using self_t       = Levelset;
  /// Defines the value type of the state data.
  using value_t      = T;
  /// Defines the type for the storage descriptor for the state. The descriptor
  /// needs to store two elements for the density and the pressure, and dims
  /// elements for the velocity componenets.
  using descriptor_t = StorageDescriptor<Layout, value_t>;
  /// Defines the type of the storage for the state.
  using storage_t    = typename descriptor_t::storage_t;

 public:
  //==--- [construction] ---------------------------------------------------==//
  
  /// Default constructor which creates the levelset.
  ripple_host_device Levelset() {}

  /// Constructor to create the levelset from the storage type.
  /// \param storage The storage to create this levelset with.
  ripple_host_device Levelset(storage_t storage) : _storage(storage) {}

  /// Constructor to create the levelset from another fluid type with a 
  /// potentially different storage layout, copying the data from the other
  /// storage to this one.
  /// \param  other       The other levelset to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device Levelset(const Levelset<T, Dims, OtherLayout>& other)
  : _storage(other._storage) {}

  /// Constructor to create the levelset from another fluid type with a 
  /// potentially different storage layout, moving the data from the other 
  /// storage to this one.
  /// \param  other       The other fluid state to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device Levelset(Levelset<T, Dims, OtherLayout>&& other)
  : _storage(std::move(other._storage)) {}
  
  //==--- [operator overload] ----------------------------------------------==//
  
  /// Overload of copy assignment operator to set the levelset from another
  /// levelset of a potentially different layout type.
  /// \param  other       The other levelset to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device auto operator=(const Levelset<T, Dims, OtherLayout>& other)
  -> self_t& {
    _storage = other._storage;
    return *this;
  }

  /// Overload of move  assignment operator to set the levelest from another 
  /// levelset of a potentially different layout type.
  /// \param  other       The other levelset to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  template <typename OtherLayout>
  ripple_host_device auto operator=(Levelset<T, Dims, OtherLayout>&& other)
  -> self_t& {
    _storage = other._storage;
    return *this;
  }

  /// Overload of move  assignment operator to set the levelest from another 
  /// levelset of a potentially different layout type.
  /// \param  other       The other levelset to create this one from.
  /// \tparam OtherLayout The storage layout of the other type.
  ripple_host_device auto operator=(T value) -> void {
    _storage.template get<0>() = value;
  }

  /// Overload of operator[] to enable array functionality on the state. This
  /// returns a reference to the \p ith stored element.
  /// \param i The index of the element to return.
  ripple_host_device auto operator[](size_t i) -> value_t& {
    return _storage.template get<0>();
  }

  /// Overload of operator[] to enable array functionality on the state. This
  /// returns a constant reference to the \p ith stored element.
  /// \param i The index of the element to return.
  ripple_host_device auto operator[](size_t i) const -> const value_t& {
    return _storage.template get<0, 0>();
  }

  //==--- [dimensions] -----------------------------------------------------==//
  
  /// Returns the number of spatial dimensions for the state.
  ripple_host_device constexpr auto dimensions() const -> size_t {
    return dims;
  }

  //==--- [interface] ------------------------------------------------------==//
  
  /// Returns a const reference to the value of the levelset.
  ripple_host_device constexpr auto value() const -> const value_t& {
    return _storage.template get<0>();
  }

  /// Returns a reference to the value of the levelset.
  ripple_host_device constexpr auto value() -> value_t& {
    return _storage.template get<0>();
  }

  /// Returns true if the value of the levelset defines it to be inside, which
  /// is taken to include the boundary (zero level set).
  ripple_host_device constexpr auto inside() const -> bool {
    return value() <= boundary_value;
  }

  /// Returns true if the value of the levelset defines it to be outside, which
  /// is defined to be completely outside of the levelset (the boundary is taken
  /// to be inside).
  ripple_host_device constexpr auto outside() const -> bool {
    return !inside();
  }

 private:
  storage_t _storage; //!< Storage for the levelset data.
};

} // namespace ripple::fv

#endif // RIPPLE_LEVELSET_LEVELSET_HPP

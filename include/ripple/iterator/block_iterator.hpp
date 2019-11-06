//==--- ripple/container/block_iterator.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  block_iterator.hpp
/// \brief This file imlements an iterator over a block.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_BLOCK_ITERATOR_HPP
#define RIPPLE_CONTAINER_BLOCK_ITERATOR_HPP

#include <ripple/storage/storage_traits.hpp>

namespace ripple {

/// The BlockIterator class defines a iterator over a block, for a given space
/// which defines the region of the block.
///
/// The type T for the iterator can be either a normal type, or type which
/// implements the StridableLayout interface. Regardless, the use is the same,
/// and the iterator operator as if it was a pointer to T.
///
/// \tparam T     The data type which the iterator will access.
/// \tparam Space The type which defines the iteration space.
template <typename T, typename Space>
class BlockIterator {
  //==--- [traits] ---------------------------------------------------------==//

  /// Defines the layout traits for the type t.
  using layout_traits_t = layout_traits_t<T>;
  /// Defines the value type for the iterator.
  using value_t         = typename layout_traits_t::value_t;
  /// Defines a reference type for the iterator.
  using ref_t           = value_t&;
  /// Defines a const reference type for the iterator.
  using const_ref_t     = const value_t&;
  /// Defines a pointer type for the iterator.
  using ptr_t           = value_t*;
  /// Defines a const pointer type for the iterator.
  using const_ptr_t     = const value_t*;
  /// Defines the type of the space for the iterator.
  using space_t         = Space;

  /// Defines the type of the storage. If the type is stridable, the access
  /// types are actual types which behave like pointers, so we store the actual
  /// type, however, for non strided types, they are normal types and therefore
  /// a pointer to the type need to be stored.
  using storage_t       =
    std::conditional_t<layout_traits_t::is_stridable_layout, value_t, value_t*>;

  //==--- [constants] ------------------------------------------------------==//

  /// Defines an overload instance for overloading implementations based on the
  /// stridability of the type T.
  static constexpr auto is_stridable_overload_v = 
    StridableOverloader<layout_traits_t::is_stridable_layout>{};

  //==--- [deref impl] -----------------------------------------------------==//

  /// Implementation of dereferencing for stridable types. Since the stridable
  /// type stores a pointer like wrapper, a reference to this type is returned.
  ripple_host_device auto deref_impl(stridable_overload_t) -> ref_t {
    return _data_ptr;
  }
  /// Implementation of dereferencing for stridable types. Since the stridable
  /// type stores a pointer like wrapper, a constant reference to this type is
  /// returned.
  ripple_host_device auto deref_impl(stridable_overload_t) const
  -> const_ref_t {
    return _data_ptr;
  }

  /// Implementation of dereferencing for non stridable types. Since for regular
  /// types the iterator stores a pointer to the type, dereferencing is required
  /// here.
  ripple_host_device auto deref_impl(non_stridable_overload_t) -> ref_t {
    return *_data_ptr;
  }
  /// Implementation of dereferencing for non stridable types. Since for regular
  /// types the iterator stores a pointer to the type, dereferencing is required
  /// here.
  ripple_host_device auto deref_impl(non_stridable_overload_t) const
  -> const_ref_t {
    return *_data_ptr;
  }

  //==--- [access impl] ----------------------------------------------------==//

  /// Implementation of accessing for stridable types. For a stirdable type, a
  /// pointer like wrapper is stored, rather than a pointer, so the address of
  /// the wrapping type needs to be returned.
  ripple_host_device auto access_impl(stridable_overload_t) -> ptr_t {
    return &_data_ptr;
  }
  /// Implementation of accessing for stridable types. For a stirdable type, a
  /// pointer like wrapper is stored, rather than a pointer, so the constant
  /// address of the wrapping type needs to be returned.
  ripple_host_device auto access_impl(stridable_overload_t) const
  -> const_ptr_t {
    return &_data_ptr;
  }
  
  /// Implementation of accessing for non stridable types. For a non stridable
  /// type, a pointer is stored, so this can just be returned without taking the
  /// address.
  ripple_host_device auto access_impl(non_stridable_overload_t) -> ptr_t {
    return _data_ptr;
  }
  /// Implementation of accessing for non stridable types. For a non stridable
  /// type, a pointer is stored, so this can just be returned without taking the
  /// address.
  ripple_host_device auto access_impl(non_stridable_overload_t) const
  -> const_ptr_t {
    return _data_ptr;
  }

  storage_t _data_ptr;  //!< A pointer to the data.
  space_t   _space;     //!< The space over which to iterate.

 public:
  /// Constructor to create the iterator from the storage type and a space over
  /// which the iterator can iterate. If the type T is a StridableLayout type,
  /// then the storage must be an implementation of the StorageAccessor
  /// interface, otherwise (for regular types) the storage must be a pointer to
  /// the type.
  /// \param data_ptr A pointer (or type which points) to the data.
  /// \param space    The space over which the iterator can iterate. 
  ripple_host_device BlockIterator(storage_t data_ptr, space_t space)
  : _data_ptr{data_ptr}, _space{space} {}

  //==--- [operator overloading] -------------------------------------------==//

  /// Overload of the dereference operator to access the type T pointed to by
  /// the iterator. This returns a reference to the type stored in the iterator.
  ripple_host_device auto operator*() -> ref_t {
    return deref_impl(is_stridable_overload_v);
  }
  /// Overload of the dereference operator to access the type T pointed to by
  /// the iterator. This returns a const reference to the type T pointer to by
  /// the iterator.
  ripple_host_device auto operator*() const -> const_ref_t {
    return deref_impl(is_stridable_overload_v);
  }

  /// Overload of the access operator to return a pointer, or pointer-like
  /// object for the type T.
  ripple_host_device auto operator->() -> ptr_t {
    return access_impl(is_stridable_overload_v);
  }

  /// Overload of the access operator to return a pointer, or pointer-like
  /// object for the type T.
  ripple_host_device auto operator->() const -> const_ptr_t {
    return access_impl(is_stridable_overload_v);
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BOX_ITERATOR_HPP

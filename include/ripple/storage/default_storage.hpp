//==--- ripple/storage/default_allocator.hpp --------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  default_allocator.hpp
/// \brief This file defines an allocation class to allocate types which do not
///        have auto layout.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_DEFAULT_STORAGE_HPP
#define RIPPLE_STORAGE_DEFAULT_STORAGE_HPP

#include <ripple/multidim/multidim_space.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/// Default allocator to provide allocation information for a type T.
/// \tparam T The type for allocation.
template <typename T>
class DefaultStorage {
  /// The pointer to the storage.
  using ptr_t     = T*;
  /// The type of the storage.
  using storage_t = DefaultStorage<T>;

  /// Allocator for default storage. This can be used to determine the memory
  /// requirement for the storage for different spatial configurations, as well
  /// as to offset DefaultStorage elements within the allocated space.
  struct Allocator {
    /// Returns the number of bytes required to allocate a total of \p elements
    /// of the types defined by Ts.
    /// \param elements The number of elements to allocate.
    ripple_host_device static constexpr auto allocation_size(
      std::size_t elements
    ) -> std::size_t {
      return sizeof(T) * elements;
    }

    /// Returns the number of bytes required to allocate a total of Elements
    /// of the types defined by Ts. This overload of the function can be used to
    /// allocate static memory when the number of elements in the space is known
    /// at compile time.
    /// \tparam Elements The number of elements to allocate.
    template <std::size_t Elements>
    ripple_host_device static constexpr auto allocation_size() -> std::size_t {
      return sizeof(T) * Elements;
    }

    /// Offsets the storage by the amount specified by the indices \p is. This
    /// assumes that the data into which the storage can offset is valid, which
    /// is the case if the storage was created through the allocator.
    ///
    /// This returns a new DefaultStorage<T>, offset to the new indices in the
    /// space.
    ///
    /// \param  storage   The storage to offset.
    /// \param  space     The space for which the storage is defined.
    /// \param  is        The indices to offset to in the space.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    /// \tparam Indices   The types of the indices.
    template <typename SpaceImpl, typename... Indices>
    ripple_host_device static auto offset(
      const storage_t&                storage,
      const MultidimSpace<SpaceImpl>& space  ,
      Indices&&...                    is
    ) -> storage_t {
      storage_t r;
      r._data = static_cast<T*>(
        storage._data + offset_to_aos(space, 1, std::forward<Indices>(is)...)
      );
      return r;
    }

    /// Creates the storage, initializing a T instance which points to \p ptr.
    /// \param  ptr       A pointer to the beginning of the memory space.
    /// \param  space     The multidimensional space which defines the domain.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    template <typename SpaceImpl>
    ripple_host_device static auto create(
      void* ptr,
      const MultidimSpace<SpaceImpl>& space
    ) -> T {
      storage_t r;
      r._data = ptr;
      return r;
    }
  };

  ptr_t _data = nullptr; //!< Pointer for the data.

 public:
  /// Defines the type of the allocator for the storage.
  using allocator_t = Allocator;
};

} // namespace ripple

#endif // RIPPLE_STORAGE_DEFAULT_STORAGE_HPP

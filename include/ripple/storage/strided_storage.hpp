//==--- ripple/storage/strided_storage.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  strided_storage.hpp
/// \brief This file implements a storage class for storing data in a strided
///        manner.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STRIDED_STORAGE_HPP
#define RIPPLE_STORAGE_STRIDED_STORAGE_HPP

#include "storage_traits.hpp"
#include <ripple/multidim/offset_to.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple {

/// This creates storage for the Ts types defined by the arguments, and it
/// creates the storage as pointers to arrays of each of the types Ts.
/// \tparam Ts The types to create storage for.
template <typename... Ts>
class StridedStorage {
  /// Defines the type of the pointer to the data.
  using ptr_t     = void*;
  /// Defines the type of the storage.
  using storage_t = StridedStorage<Ts...>;

  //==--- [traits] ---------------------------------------------------------==//

  /// Gets the value type of the storage element traits for a type.
  /// \tparam T The type to get the storage element traits for.
  template <typename T>
  using element_value_t = typename storage_element_traits_t<T>::value_t;

  /// Gets the value type of the storage element traits the type at position I.
  /// \tparam I The index of the type to get the value type for.
  template <std::size_t I>
  using element_at_value_t = element_value_t<
    std::tuple_element_t<I, std::tuple<Ts...>>
  >;

  //==--- [constants] ------------------------------------------------------==//
  
  /// Returns the number of different types.
  static constexpr auto num_types = sizeof...(Ts);

  /// Gets the numbber of components for the storage element.
  /// \tparam T The type to get the size of.
  template <typename T>
  static constexpr auto element_components_v =
    storage_element_traits_t<T>::num_elements;

  /// Defines the effective byte size of all elements to store.
  static constexpr auto storage_byte_size_v =
    (storage_element_traits_t<Ts>::byte_size + ... + std::size_t{0});

  //==--- [size containers] ------------------------------------------------==//
  
  /// Defines the sizes of each of the types.
  static constexpr std::size_t byte_sizes[num_types] = {
    sizeof(element_value_t<Ts>)...
  };
  /// Defines the number of components in each of the types.
  static constexpr std::size_t components[num_types] = {
    element_components_v<Ts>...
  };

  //==--- [allocator] ------------------------------------------------------==//

  /// Allocator for the strided storage. This can be used to determine the
  /// memory requirement for the storage for a spicifc spatial configuration, as
  /// well as to access into the storage space.
  struct Allocator {
    /// Returns the number of bytes required to allocate a total of \p elements
    /// of the types defined by Ts.
    ///
    /// \param elements The number of elements to allocate.
    ripple_host_device static constexpr auto allocation_size(
      std::size_t elements
    ) -> std::size_t {
      return storage_byte_size_v * elements;
    }

    /// Returns the number of bytes required to allocate a total of Elements
    /// of the types defined by Ts. This overload of the function can be used to
    /// allocate static memory when the number of elements in the space is known
    /// at compile time.
    ///
    /// \tparam Elements The number of elements to allocate.
    template <std::size_t Elements>
    ripple_host_device static constexpr auto allocation_size() -> std::size_t {
      return storage_byte_size_v * Elements;
    }

    /// Creates the storage, initializing a StridedStorage instance which has
    /// its data pointers pointing to the correc locations in the memory space
    /// which is pointed to by \p ptr. The memory space should have a size which
    /// is returned by the `allocation_size()` method, otherwise this may index
    /// into undefined memory.
    /// \param  ptr       A pointer to the beginning of the memory space.
    /// \param  space     The multidimensional space which defines the domain.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    template <typename SpaceImpl>
    ripple_host_device static auto create(
      void* ptr,
      const MultidimSpace<SpaceImpl>& space
    ) -> storage_t {
      storage_t r;
      r._stride       = space.size(dim_x);
      r._data[0]      = ptr;
      const auto size = space.size();
      auto offset     = 0;
      unrolled_for<num_types - 1>([&] (auto prev_index) {
        constexpr auto curr_index = prev_index + 1;
        offset += components[prev_index] * size * byte_sizes[prev_index];
        r._data[curr_index] = static_cast<void*>(
          static_cast<char*>(ptr) + offset
        );
      });
      return r;
    }

    /// Offsets the storage by the amount specified by the indices \p is. This
    /// assumes that the data into which the storage can offset is valid, which
    /// is the case if the storage was created through the allocator.
    ///
    /// This returns a new StridedStorage, offset to the new indices in the
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
      r._stride = storage._stride;
      unrolled_for<num_types>([&] (auto i) {
        using type_t = element_at_value_t<i>;
        r._data[i]   = static_cast<void*>(
          static_cast<type_t*>(storage._data[i]) + 
          offset_to_soa(space, components[i], std::forward<Indices>(is)...)
        );
      });
      return r;
    }
  };

  //==--- [members] --------------------------------------------------------==//

  ptr_t     _data[num_types]; //!< Pointers to the data.
  uint32_t  _stride = 1;      //!< Stride between elements.

 public:
  //==--- [traits] ---------------------------------------------------------==//

  /// Defines the type of the allocator for creating StridedStorage.
  using allocator_t = Allocator;

  //==--- [interface] ------------------------------------------------------==//

  /// Gets a reference to the Jth element of the Ith data type. This will only
  /// be enabled when the type of the Ith type is a StorageElement<> so that the
  /// call to operator[] on the Ith type is valid.
  /// \tparam I The index of the type to get the data from.
  /// \tparam J The index in the type to get.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    std::size_t J,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    storage_element_enable_t<T> = 0
  >
  auto get() -> element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!"
    );
    return static_cast<element_value_t<T>*>(_data[I])[J * _stride];
  }

  /// Gets a const reference to the Jth element of the Ith data type. This will
  /// only be enabled when the type of the Ith type is a StorageElement<> so
  /// that the call to operator[] on the Ith type is valid.
  /// \tparam I The index of the type to get the data from.
  /// \tparam J The index in the type to get.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    std::size_t J,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    storage_element_enable_t<T> = 0
  >
  auto get() const -> const element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!"
    );
    return static_cast<const element_value_t<T>*>(_data[I])[J * _stride]; 
  }

  /// Gets a reference to the Ith data type. This will only be enabled when the
  /// type of the Ith type is not a StorageElement<>.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    non_storage_element_enable_t<T> = 0
  >
  auto get() -> element_value_t<T>& {
    return *static_cast<element_value_t<T>*>(_data[I]);
  }

  /// Gets a const reference to the Ith data type. This will only be enabled
  /// when the type of the Ith type is not a StorageElement<>.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = std::tuple_element_t<I, std::tuple<Ts...>>,
    non_storage_element_enable_t<T> = 0
  >
  auto get() const -> const element_value_t<T>& {
    return *static_cast<const element_value_t<T>*>(_data[I]); 
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_STRIDED_STORAGE_HPP

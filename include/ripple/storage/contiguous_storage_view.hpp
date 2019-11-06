//==--- ripple/storage/contiguous_storage_view.hpp --------- -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  contiguous_storage_view.hpp
/// \brief This file implements a storage class for storing data in a contiguous
///        manner.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_CONTIGUOUS_STORAGE_VIEW_HPP
#define RIPPLE_STORAGE_CONTIGUOUS_STORAGE_VIEW_HPP

#include "storage_traits.hpp"
#include "storage_accessor.hpp"

namespace ripple {

/// Defines a view into contiguous storage for Ts types.
/// See ContiguousStorageView for more information.
/// \tparam Ts The types to create a storage view for.
template <typename... Ts>
class ContiguousStorageView : 
    public StorageAccessor<ContiguousStorageView<Ts...>> {
  /// Defines the data type for the buffer.
  using ptr_t           = void*;
  /// Defines the type of the storage.
  using storage_t       = ContiguousStorageView<Ts...>;
  /// Defines the type of owned storage.
  using owned_storage_t = OwnedStorage<Ts...>;

  //==--- [traits] ---------------------------------------------------------==//

  /// Gets the value type of the storage element traits for a type.
  /// \tparam T The type to get the storage element traits for.
  template <typename T>
  using element_value_t = typename storage_element_traits_t<T>::value_t;

  //==--- [constants] ------------------------------------------------------==//

  /// Defines the number of different types.
  static constexpr auto num_types = sizeof...(Ts);

  /// Gets the numbber of components for the storage element.
  /// \tparam T The type to get the size of.
  template <typename T>
  static constexpr auto element_components_v =
    storage_element_traits_t<T>::num_elements;

  /// Defines the sizes of each of the types.
  static constexpr std::size_t byte_sizes[num_types] = {
    storage_element_traits_t<Ts>::byte_size...
  };
  /// Defines the alignment sizes of each of the types.
  static constexpr std::size_t align_sizes[num_types] = {
    storage_element_traits_t<Ts>::align_size...
  };
  /// Defines the number of components in each of the types.
  static constexpr std::size_t components[num_types] = {
    element_components_v<Ts>...
  };

  /// Returns the effective byte size of all elements to store, including any
  /// required padding. This should not be called, other than to define
  /// byte_size_v, since it is expensive to compile.
  static constexpr auto storage_byte_size() -> std::size_t {
    auto size = byte_sizes[0];
    for (size_t i = 1; i < num_types; ++i) {
      // If not aligned, find the first alignment after the total size:
      if ((size % align_sizes[i]) != 0) {
        auto first_align = align_sizes[i];
        while (first_align < size) {
          first_align += align_sizes[i];
        }
        // New size is smallest alignment.
        size = first_align;
      }
      // Add the size of the component:
      size += byte_sizes[i];
    }
    return size;
  }

  /// Returns the offset, in bytes, to the index I in the type list, accounding
  /// for any required padding dues to __badly specifier__ data layout.
  /// \tparam I The index of the type to get the offset to.
  template <std::size_t I>
  static constexpr auto offset_to() {
    auto offset = std::size_t{0};
    for (std::size_t i = 1; i <= I; ++i) {
      offset += byte_sizes[i - 1];
      // If not aligned, find the first alignment after the total size:
      if ((offset % align_sizes[i]) != 0 || offset < align_sizes[i]) {
        auto first_align = align_sizes[i];
        while (first_align < offset) {
          first_align += align_sizes[i];
        }
        // New size is smallest alignment.
        offset = first_align;
      }
    }
    return offset;
  }

  /// Returns the effective byte size of all elements in the storage.
  static constexpr auto storage_byte_size_v = storage_byte_size();

  //==--- [allocator] ------------------------------------------------------==//

  /// Allocator for contiguous storage. This can be used to determine the memory
  /// requirement for the storage for different spatial configurations, as well
  /// as to offset ContiguousStorage elements within the allocated space.
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

    /// Offsets the storage by the amount specified by the indices \p is. This
    /// assumes that the data into which the storage can offset is valid, which
    /// is the case if the storage was created through the allocator.
    ///
    /// This returns a new ContiguousStorage, offset to the new indices in the
    /// space.
    ///
    /// \param  storage   The storage to offset.
    /// \param  space     The space for which the storage is defined.
    /// \param  is        The indices to offset to in the space.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    /// \tparam Indices   The types of the indices.
    template <
      typename SpaceImpl,
      typename... Indices, variadic_ge_enable_t<1, Indices...> = 0
    >
    ripple_host_device static auto offset(
      const storage_t&                storage,
      const MultidimSpace<SpaceImpl>& space  ,
      Indices&&...                    is
    ) -> storage_t {
      storage_t r;
      r._data = static_cast<void*>(
        static_cast<char*>(storage._data) + offset_to_aos(
          space, storage_byte_size_v, std::forward<Indices>(is)...
        )
      );
      return r;
    }

    /// Creates the storage, initializing a ContiguousStorage instance which has
    /// its data pointer pointing to the location defined by the indices \p is.
    /// which is pointed to by \p ptr. The memory space should have a size which
    /// is returned by the `allocation_size()` method, otherwise this may index
    /// into undefined memory.
    /// \param  ptr       A pointer to the beginning of the memory space.
    /// \param  space     The multidimensional space which defines the domain.
    /// \param  is        The indices of the element in the space to create.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    /// \tparam Indices   The type of the indices.
    template <
      typename    SpaceImpl,
      typename... Indices  , variadic_ge_enable_t<1, Indices...> = 0
    > 
    ripple_host_device static auto create(
      void*                           ptr  ,
      const MultidimSpace<SpaceImpl>& space,
      Indices&&...                    is
    ) -> storage_t {
      storage_t r;
      r._data = static_cast<void*>(
        static_cast<char*>(ptr) + offset_to_aos(
          space, storage_byte_size_v, std::forward<Indices>(is)...  
        )  
      );
      return r;
    }

    /// Creates the storage, initializing a ContiguousStorage instance which has
    /// its data pointer pointing to the \p ptr. The memory space should have a
    /// size which is returned by the `allocation_size()` method, otherwise this
    /// may index into undefined memory.
    /// \param  ptr       A pointer to the beginning of the memory space.
    /// \param  space     The multidimensional space which defines the domain.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    template <typename SpaceImpl> 
    ripple_host_device static auto create(
      void*                           ptr,
      const MultidimSpace<SpaceImpl>& space
    ) -> storage_t {
      storage_t r;
      r._data = ptr;
      return r;
    }
  };

  //==--- [members] --------------------------------------------------------==//

  ptr_t _data = nullptr;  //!< Pointer to the data.

 public:
  //==--- [traits] ---------------------------------------------------------==//

  /// Defines the type of the allocator for creating StridedStorage.
  using allocator_t = Allocator;

  //==--- [operator overload] ----------------------------------------------==//

  /// Overload of operator= to set the data for the ContiguousStorageView from
  /// another StorageAccessor. This returns the contiguousStorageView with the
  /// data copied from \p from.
  /// \param  from The accessor to copy the data from.
  /// \tparam Impl The implementation of the StorageAccessor.
  template <typename Impl>
  ripple_host_device auto operator=(const StorageAccessor<Impl>& from)
  -> storage_t& {
    unrolled_for<num_types>([&] (auto i) {
      constexpr std::size_t type_idx = i;
      constexpr auto        values   = 
        element_components_v<nth_element_t<type_idx, Ts...>>;

      copy_from_to<type_idx, values>(from, *this);
    });
    return *this;
  }

  //==--- [interface] ------------------------------------------------------==//

  /// Returns the number of components in the Ith type being stored. For
  /// non-indexable types this will always return 1, otherwise will return the
  /// number of possible components which can be indexed.
  /// \tparam I The index of the type to get the number of components for.
  template <std::size_t I>
  ripple_host_device constexpr auto components_of() const -> std::size_t {
    return components[I];
  }

  /// Gets a reference to the Ith data type. This will only be enabled when the
  /// type of the Ith type is not a StorageElement<>.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = nth_element_t<I, Ts...>,
    non_storage_element_enable_t<T> = 0
  >
  auto get() -> element_value_t<T>& {
    constexpr auto offset = offset_to<I>();
    return *reinterpret_cast<element_value_t<T>*>(
      static_cast<char*>(_data) + offset
    );
  }

  /// Gets a const reference to the Ith data type. This will only be enabled
  /// when the type of the Ith type is not a StorageElement<>.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = nth_element_t<I, Ts...>,
    non_storage_element_enable_t<T> = 0
  >
  auto get() const -> const element_value_t<T>& {
    constexpr auto offset = offset_to<I>();
    return *reinterpret_cast<const element_value_t<T>*>(
      static_cast<const char*>(_data) + offset
    );
  }

  /// Gets a reference to the Jth element of the Ith data type. This will only
  /// be enabled when the type of the Ith type is a StorageElement<> so that the
  /// call to operator[] on the Ith type is valid.
  /// \tparam I The index of the type to get the data from.
  /// \tparam J The index in the type to get.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    std::size_t J,
    typename    T = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0
  >
  auto get() -> element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!"
    );
    constexpr auto offset = offset_to<I>();
    return reinterpret_cast<element_value_t<T>*>(
      static_cast<char*>(_data) + offset
    )[J];
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
    typename    T = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0
  >
  auto get() const -> const element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!"
    );
    constexpr auto offset = offset_to<I>();
    return reinterpret_cast<const element_value_t<T>*>(
      static_cast<const char*>(_data) + offset
    )[J];
  }

  /// Gets a reference to the jth element of the Ith data type. This will only
  /// be enabled when the type of the Ith type is a StorageElement<> so that the
  /// call to operator[] on the Ith type is valid.
  /// \param  j The index of the component in the type to get.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0
  >
  auto get(std::size_t j) -> element_value_t<T>& {
    constexpr auto offset = offset_to<I>();
    return reinterpret_cast<element_value_t<T>*>(
      static_cast<char*>(_data) + offset
    )[j];
  }

  /// Gets a const reference to the jth element of the Ith data type. This will
  /// only be enabled when the type of the Ith type is a StorageElement<> so
  /// that the call to operator[] on the Ith type is valid.
  /// \param  j The index of the component in the type to get.
  /// \tparam I The index of the type to get the data from.
  /// \tparam T The type of the Ith element.
  template <
    std::size_t I,
    typename    T = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0
  >
  auto get(std::size_t j) const -> const element_value_t<T>& {
    constexpr auto offset = offset_to<I>();
    return reinterpret_cast<const element_value_t<T>*>(
      static_cast<const char*>(_data) + offset
    )[j];
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_CONTIGUOUS_STORAGE_VIEW_HPP
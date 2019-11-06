//==--- ripple/storage/strided_storage_view.hpp ------------ -*- C++ -*- ---==//
//            
//                                Ripple
// 
//                      Copyright (c) 2019 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  strided_storage_view.hpp
/// \brief This file implements a storage class which views data which is
///        strided (SoA), and which knows how to allocate and offset into such
///        data.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_STORAGE_STRIDED_STORAGE_VIEW_HPP
#define RIPPLE_STORAGE_STRIDED_STORAGE_VIEW_HPP

#include "storage_traits.hpp"
#include "storage_accessor.hpp"
#include <ripple/multidim/offset_to.hpp>
#include <ripple/utility/dim.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple {

/// Defines a view into strided storage for Ts types.
/// See StridedStorageView for more information.
/// \tparam Ts The types to create a storage view for.
template <typename... Ts>
class StridedStorageView : public StorageAccessor<StridedStorageView<Ts...>> {
  /// Defines the type of the pointer to the data.
  using ptr_t     = void*;
  /// Defines the type of the storage.
  using storage_t = StridedStorageView<Ts...>;

  //==--- [traits] ---------------------------------------------------------==//

  /// Gets the value type of the storage element traits for a type.
  /// \tparam T The type to get the storage element traits for.
  template <typename T>
  using element_value_t = typename storage_element_traits_t<T>::value_t;

  /// Gets the value type of the storage element traits the type at position I.
  /// \tparam I The index of the type to get the value type for.
  template <std::size_t I>
  using nth_element_value_t = element_value_t<nth_element_t<I, Ts...>>;

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
   private:
    /// Returns the scaling factor when offsetting in the y dimenion.
    /// \param  dim The dimension to base the scaling on.
    /// \tparam I   The index of the component to get the scaling factor from.
    template <std::size_t I>
    static constexpr auto offset_scale(Num<I>, dimx_t) {
      return 1;
    }

    /// Returns the scaling factor when offsetting in the y dimenion.
    /// \param  dim The dimension to base the scaling on.
    /// \tparam I   The index of the component to get the scaling factor from.
    template <std::size_t I>
    static constexpr auto offset_scale(Num<I>, dimy_t dim) {
      return components[I];
    }

    /// Returns the scaling factor when offsetting in the z dimenion.
    /// \param  dim The dimension to base the scaling on.
    /// \tparam I   The index of the component to get the scaling factor from.
    template <std::size_t I>
    static constexpr auto offset_scale(Num<I>, dimz_t dim) {
      return components[I];
    }
  
    /// Returns the scaling factor when offsetting with a dimension which is a
    /// size type.
    /// \param  dim The dimension to base the scaling on.
    /// \tparam I   The index of the component to get the scaling factor from.
    template <std::size_t I>
    static constexpr auto offset_scale(Num<I>, std::size_t dim) -> std::size_t {
      return dim == 0 ? 1 : components[I];
    }

   public:
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
    /// This returns a new StridedStorage, offset to the new indices in the
    /// space.
    ///
    /// \param  storage   The storage to offset.
    /// \param  space     The space for which the storage is defined.
    /// \param  is        The indices to offset to in the space.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    /// \tparam Indices   The types of the indices.
    template <
      typename    SpaceImpl,
      typename... Indices  , variadic_ge_enable_t<1, Indices...> = 0
    >
    ripple_host_device static auto offset(
      const storage_t&                storage,
      const MultidimSpace<SpaceImpl>& space,
      Indices&&...                    is
    ) -> storage_t {
      storage_t r;
      r._stride = storage._stride;
      unrolled_for<num_types>([&] (auto i) {
        using type_t = nth_element_value_t<i>;
        r._data[i]   = static_cast<void*>(
          static_cast<type_t*>(storage._data[i]) + 
          offset_to_soa(space, components[i], std::forward<Indices>(is)...)
        );
      });
      return r;
    }

    /// Offsets the storage by the amount specified by \p amount in the
    /// dimension \p dim.
    ///
    /// This returns a new StridedStorage, offset to the new indices in the
    /// space.
    ///
    /// \param  storage   The storage to offset.
    /// \param  space     The space for which the storage is defined.
    /// \param  dim       The dimension to offset in.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    /// \tparam Dim       The type of the dimension.
    template <typename SpaceImpl, typename Dim, diff_enable_t<Dim, int> = 0>
    ripple_host_device static auto offset(
      const storage_t&                storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount
    ) -> storage_t {
      storage_t r;
      r._stride = storage._stride;
      unrolled_for<num_types>([&] (auto i) {
        using type_t = nth_element_value_t<i>;
        r._data[i]   = static_cast<void*>(
          static_cast<type_t*>(storage._data[i]) + 
          amount * space.step(dim) * offset_scale(i, dim)
        );
      });
      return r;
    }

    /// Shifts the storage by the amount specified by \p amount in the
    /// dimension \p dim.
    ///
    /// This returns a new StridedStorage, offset to the new indices in the
    /// space.
    ///
    /// \param  storage   The storage to offset.
    /// \param  space     The space for which the storage is defined.
    /// \param  dim       The dimension to offset in.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    /// \tparam Dim       The type of the dimension.
    template <typename SpaceImpl, typename Dim>
    ripple_host_device static auto shift(
      storage_t&                      storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount
    ) -> void {
      unrolled_for<num_types>([&] (auto i) {
        using type_t     = nth_element_value_t<i>;
        storage._data[i] = static_cast<void*>(
          static_cast<type_t*>(storage._data[i]) + 
          amount * space.step(dim) * offset_scale(i, dim)
        );
      });
    }

    /// Creates the storage, initializing a StridedStorage instance which has
    /// its data pointers pointing to the \p ptr, and then offset by the \p is
    /// amounts in the memory space. The memory space should have a size which
    /// is returned by the `allocation_size()` method, otherwise this may index
    /// into undefined memory. This returns a new StridedStorage, offset to the
    /// new indices in the space.
    ///
    /// \param  ptr       A pointer to the data to create the storage in.
    /// \param  space     The space for which the storage is defined.
    /// \param  is        The indices to offset to in the space.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    /// \tparam Indices   The types of the indices.
    template <
      typename    SpaceImpl,
      typename... Indices  , variadic_ge_enable_t<1, Indices...> = 0
    >
    ripple_host_device static auto create(
      void*                           ptr    ,
      const MultidimSpace<SpaceImpl>& space  ,
      Indices&&...                    is
    ) -> storage_t {
      storage_t r = create(ptr, space);
      return offset(r, space, is...);
    }

    /// Creates the storage, initializing a StridedStorage instance which has
    /// its data pointers pointing to the \p ptr. The memory space should have
    /// a size which is returned by the `allocation_size()` method, otherwise
    /// this may index into undefined memory.
    /// \param  ptr       A pointer to the beginning of the memory space.
    /// \param  space     The multidimensional space which defines the domain.
    /// \tparam SpaceImpl The implementation of the spatial interface.
    template <typename SpaceImpl>
    ripple_host_device static auto create(
      void*                           ptr,
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
  };

  //==--- [members] --------------------------------------------------------==//

  ptr_t     _data[num_types]; //!< Pointers to the data.
  uint32_t  _stride = 1;      //!< Stride between elements.

 public:
  //==--- [traits] ---------------------------------------------------------==//

  /// Defines the type of the allocator for creating StridedStorage.
  using allocator_t = Allocator;

  //==--- [construction] ---------------------------------------------------==//

  /// Default constructor for the strided storage.
  ripple_host_device StridedStorageView() = default;

  /// Constructor to set the strided storage from another StorageAccessor.
  /// \param  from The accessor to copy the data from.
  /// \tparam Impl The implementation of the StorageAccessor.
  template <typename Impl>
  ripple_host_device StridedStorageView(const StorageAccessor<Impl>& from) {
    unrolled_for<num_types>([&] (auto i) {
      constexpr std::size_t type_idx = i;
      constexpr auto        values   = 
        element_components_v<nth_element_t<type_idx, Ts...>>;

      copy_from_to<type_idx, values>(from, *this);
    });
  }

  //==--- [operator overload] ----------------------------------------------==//
  
  /// Overload of operator= to set the data for the StridedStorageView from
  /// another StorageAccessor. This returns the StridedStorageView with the
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
    return *static_cast<element_value_t<T>*>(_data[I]);
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
    return *static_cast<const element_value_t<T>*>(_data[I]); 
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
    typename    T = nth_element_t<I, Ts...>,
    storage_element_enable_t<T> = 0
  >
  auto get() const -> const element_value_t<T>& {
    static_assert(
      J < element_components_v<T>, "Out of range acess for storage element!"
    );
    return static_cast<const element_value_t<T>*>(_data[I])[J * _stride]; 
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
    return static_cast<element_value_t<T>*>(_data[I])[j * _stride];
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
    return static_cast<const element_value_t<T>*>(_data[I])[j * _stride];
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_STRIDED_STORAGE_VIEW_HPP

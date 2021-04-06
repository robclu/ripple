/**=--- ripple/storage/strided_storage_view.hpp ------------ -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  strided_storage_view.hpp
 * \brief This file implements a storage class which views data which is
 *        strided (SoA), and which knows how to allocate and offset into such
 *        data.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_STORAGE_STRIDED_STORAGE_VIEW_HPP
#define RIPPLE_STORAGE_STRIDED_STORAGE_VIEW_HPP

#include "storage_element_traits.hpp"
#include "storage_traits.hpp"
#include "storage_accessor.hpp"
#include <ripple/space/offset_to.hpp>
#include <ripple/utility/type_traits.hpp>

namespace ripple {

/**
 * Defines a view into strided storage for Ts types.
 *
 * The data for this class is stided in that each element of a multi-element
 * type his offset by the stride of the zero dimension, which is the number of
 * elements in the zero dimension, including padding.
 *
 *
 * \tparam Ts The types to create a storage view for.
 */
template <typename... Ts>
class StridedStorageView : public StorageAccessor<StridedStorageView<Ts...>> {
  // clang-format off
  /** Defines the type of the pointer to the data. */
  using Ptr      = void*;
  /** Defines the type of a const pointer to the data. */
  using ConstPtr = const void*;
  /** Defiens the type of this storage. */
  using Storage  = StridedStorageView;
  // clang-format on

  /** LayoutTraits is a friend to allow allocator access. */
  template <typename T, bool B>
  friend struct LayoutTraits;

  /*==--- [traits] ---------------------------------------------------------==*/

  /**
   * Gets the value type of the storage element traits for a type.
   * \tparam T The type to get the storage element traits for.
   */
  template <typename T>
  using element_value_t = typename storage_element_traits_t<T>::Value;

  /**
   * Gets the value type of the storage element traits the type at position I.
   * \tparam I The index of the type to get the value type for.
   */
  template <size_t I>
  using nth_element_value_t = element_value_t<nth_element_t<I, Ts...>>;

  /*==--- [constants] ------------------------------------------------------==*/

  /** Returns the number of different types. */
  static constexpr size_t num_types = sizeof...(Ts);

  /**
   * Gets the numbber of components for the storage element.
   * \tparam T The type to get the size of.
   */
  template <typename T>
  static constexpr size_t element_components =
    storage_element_traits_t<T>::num_elements;

  /** Defines the effective byte size of all elements to store. */
  static constexpr size_t storage_byte_size =
    (storage_element_traits_t<Ts>::byte_size + ... + size_t{0});

  /**
   * Gets the number of bytes required for the nth element.
   * \tparam I The index of the component to get the number of bytes for.
   */
  template <size_t I>
  static constexpr size_t nth_element_bytes = sizeof(nth_element_value_t<I>);

  /**
   * Returns the number of components for the Nth element.
   * \param I The index of the element to get the number of components of.
   */
  template <size_t I>
  static constexpr size_t nth_element_components_v =
    storage_element_traits_t<nth_element_t<I, Ts...>>::num_elements;

  /*==--- [allocator] ------------------------------------------------------==*/

  /**
   * Allocator for the strided storage. This can be used to determine the
   * memory requirement for the storage for a spicifc spatial configuration, as
   * well as to access into the storage space.
   */
  struct Allocator {
   private:
    /**
     * Returns the scaling factor when offsetting in the x dimenion.
     * \tparam I   The index of the component to get the scaling factor from.
     * \return The scaling factor for ofsetting in the x dimension.
     */
    template <size_t I>
    ripple_host_device static constexpr auto
    offset_scale(Num<I>, DimX) noexcept -> size_t {
      return 1;
    }

    /**
     * Returns the scaling factor when offsetting in the y dimenion.
     * \tparam I The index of the component to get the scaling factor from.
     * \return The scaling factor for ofsetting in the y dimension.
     */
    template <size_t I>
    ripple_host_device static constexpr auto
    offset_scale(Num<I>, DimY) noexcept -> size_t {
      return nth_element_components<I>;
    }

    /**
     * Returns the scaling factor when offsetting in the z dimenion.
     * \param  dim The dimension to base the scaling on.
     * \tparam I   The index of the component to get the scaling factor from.
     * \return The scaling factor for ofsetting in the z dimension.
     */
    template <size_t I>
    ripple_host_device static constexpr auto
    offset_scale(Num<I>, DimZ) noexcept -> size_t {
      return nth_element_components<I>;
    }

    /**
     * Returns the scaling factor when offsetting in the given dimension.
     * \param  dim The dimension to base the scaling on.
     * \tparam I   The index of the component to get the scaling factor from.
     * \return The scaling factor for offsetting in the given dimension.
     */
    template <size_t I>
    ripple_host_device static constexpr auto
    offset_scale(Num<I>, size_t dim) noexcept -> size_t {
      return dim == 0 ? 1 : nth_element_components<I>;
    }

   public:
    /**
     * Returns the alignment required to allocate the storage.
     */
    static constexpr size_t alignment =
      max_element(storage_element_traits_t<Ts>::align_size...);

    /**
     * Computes the number of bytes required to allocate the the number of
     * specified elements.
     * \param elements The number of elements to allocate.
     * \return The number of bytes required to allocate the number of elements.
     */
    ripple_host_device static constexpr auto
    allocation_size(size_t elements) noexcept -> size_t {
      return storage_byte_size * elements;
    }

    /**
     * Computesthe number of bytes required to allocate a total of Elements
     * of the types defined by Ts. This overload of the function can be used to
     * allocate static memory when the number of elements in the space is known
     * at compile time.
     *
     * \tparam Elements The number of elements to allocate.
     * \return The number of bytes required to allocated the number of elements.
     */
    template <size_t Elements>
    ripple_host_device static constexpr auto
    allocation_size() noexcept -> size_t {
      return storage_byte_size * Elements;
    }

    /**
     * Gets the number of types which are stored strided.
     * \return The number of types which are stored strided.
     */
    static constexpr auto strided_types() noexcept -> size_t {
      return num_types;
    }

    /**
     * Gets the number of elements in the Ith type.
     * \tparam I The index of the type.
     * \return The number of elements in the ith type.
     */
    template <size_t I>
    static constexpr auto num_elements() noexcept -> size_t {
      static_assert(I < num_types, "Invalid type index!");
      return nth_element_components_v<I>;
    }

    /**
     * Gets the number of bytes for an elemeent in the Ith type.
     * \tparam I The index of the type to get the element size of.
     * \return The number of bytes for an element in the Ith type.
     */
    template <size_t I>
    static constexpr auto element_byte_size() noexcept -> size_t {
      static_assert(I < num_types, "Invalid type index!");
      return nth_element_bytes<I>;
    }

    /**
     * Offsets the storage by the amount specified in the given dimension.
     *
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     * \return A new strided storage offset by the given amount.
     */
    template <typename SpaceImpl, typename Dim>
    ripple_host_device static auto offset(
      const Storage&                  storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) noexcept -> Storage {
      Storage r;
      r.stride_ = storage.stride_;
      unrolled_for<num_types>([&](auto i) {
        using Type = nth_element_value_t<i>;
        r.data_[i] = static_cast<void*>(
          static_cast<Type*>(storage.data_[i]) +
          amount * space.step(ripple_forward(dim)) *
            offset_scale(i, ripple_forward(dim)));
      });
      return r;
    }

    /**
     * Shifts the storage by the amount specified in the given dimension.
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     * \return A new strided storage offset by the given amount.
     */
    template <typename SpaceImpl, typename Dim>
    ripple_host_device static auto shift(
      Storage&                        storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) noexcept -> void {
      unrolled_for<num_types>([&](auto i) {
        using Type       = nth_element_value_t<i>;
        storage.data_[i] = static_cast<void*>(
          static_cast<Type*>(storage.data_[i]) +
          amount * space.step(ripple_forward(dim)) *
            offset_scale(i, ripple_forward(dim)));
      });
    }

    /**
     * Creates the storage, initializing a StridedStorage instance which has
     * its data pointers pointing to the given pointer.
     *
     * \note The memory space should have a size which is that returned by the
     *       `allocation_size()` method, otherwise this may index into undefined
     *       memory.
     *
     * \param  ptr       A pointer to the data to create the storage in.
     * \param  space     The space for which the storage is defined.
     * \param  is        The indices to offset to in the space.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Indices   The types of the indices.
     * \return A new strided storage pointing to given pointer.
     */
    template <typename SpaceImpl>
    ripple_host_device static auto
    create(void* ptr, const MultidimSpace<SpaceImpl>& space) noexcept
      -> Storage {
      Storage r;
      r.stride_         = space.size(dimx());
      r.data_[0]        = ptr;
      const auto size   = space.size();
      auto       offset = 0;
      unrolled_for<num_types - 1>([&](auto prev_index) {
        constexpr auto curr_index      = prev_index + 1;
        constexpr auto components_prev = nth_element_components<prev_index>;
        constexpr auto bytes_prev      = nth_element_bytes<prev_index>;
        offset += components_prev * size * bytes_prev;
        r.data_[curr_index] =
          static_cast<void*>(static_cast<char*>(ptr) + offset);
      });
      return r;
    }
  };

  /*==--- [members] --------------------------------------------------------==*/

  /*
   * NOTE: An alternative implementation would be to store a single pointer, and
   * then modify the offsetting of the data to the different types. Both
   * implementations were benchmarked on a number of different use cases and
   * there was little difference. While the (potentially additional) pointers
   * slightly increase memory use, the single pointer implementation is more
   * complex and increased register usage in gpu code.
   */

  Ptr      data_[num_types]; //!< Pointers to the data.
  uint32_t stride_ = 1;      //!< Stride between elements.

 public:
  /**
   * Gets the number of components for the nth element.
   * \tparam I The index of the element to get the number of components for.
   */
  template <size_t I>
  static constexpr size_t nth_element_components =
    element_components<nth_element_t<I, Ts...>>;

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor for the strided storage.
   */
  StridedStorageView() = default;

  /**
   * Constructor to set the strided storage from another StorageAccessor with a
   * different layout.
   *
   * \param  other The accessor to set this storage from.
   * \tparam Impl The implementation of the StorageAccessor.
   */
  template <typename Impl>
  ripple_host_device
  StridedStorageView(const StorageAccessor<Impl>& other) noexcept {
    copy(static_cast<const Impl&>(other));
  }

  /**
   * Copy constructor to set the strided storage from the other storage.
   * \param other The other storage to set this one from.
   */
  ripple_host_device
  StridedStorageView(const StridedStorageView& other) noexcept
  : stride_{other.stride_} {
    unrolled_for<num_types>([&](auto i) { data_[i] = other.data_[i]; });
  }

  /**
   * Move constructor to move the other storage into this one.
   * \param other The other storage to move into this one.
   */
  ripple_host_device StridedStorageView(StridedStorageView&& other) noexcept
  : stride_{other.stride_} {
    unrolled_for<num_types>([&](auto i) {
      data_[i]       = other.data_[i];
      other.data_[i] = nullptr;
    });
  }

  /*==--- [operator overload] ----------------------------------------------==*/

  /**
   * Overload of assignment operator to set the data for the
   * StridedStorageView from  another StorageAccessor. \param  other The
   * accessor to copy the data from. \tparam Impl  The implementation of the
   * StorageAccessor. \return A reference to the created storage view.
   */
  template <typename Impl>
  ripple_host_device auto operator=(const StorageAccessor<Impl>& other) noexcept
    -> StridedStorageView& {
    copy(static_cast<const Impl&>(other));
    return *this;
  }

  /**
   * Overload of assignment operator to set the data for the
   * StridedStorageView from  another StridedStorageView. \param  other The
   * strided storage to copy from. \return A reference to the created storage
   * view.
   */
  ripple_host_device auto
  operator=(const StridedStorageView& other) noexcept -> StridedStorageView& {
    copy(other);
    return *this;
  }

  /**
   * Overload of move assignment operator to move the other strided view into
   * this one.
   *
   *\param  other The strided storage to move from.
   * \return A reference to the created storage view.
   */
  ripple_host_device auto
  operator=(StridedStorageView&& other) noexcept -> StridedStorageView& {
    stride_ = other.stride_;
    unrolled_for<num_types>([&](auto i) {
      data_[i]       = other.data_[i];
      other.data_[i] = nullptr;
    });
    return *this;
  }
  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets a pointer to the data for the storage.
   * \return A pointer to the data for the storage.
   */
  ripple_host_device auto data() noexcept -> Ptr {
    return &data_[0];
  }

  /**
   * Gets a const pointer to the data.
   * \return A const pointer to the data for the storage.
   */
  ripple_host_device auto data() const noexcept -> ConstPtr {
    return &data_[0];
  }

  /**
   * Returns a reference to the data pointers for the storage.
   */
  ripple_host_device auto data_ptrs() noexcept -> std::vector<Ptr> {
    std::vector<Ptr> p;
    unrolled_for<num_types>([&](auto i) { p.push_back(data_[i]); });
    return p;
  }

  /**
   * Copies the data from the other type.
   *
   * \note If the other type is not a StorageAccessor, this will cause a
   * compile time error.
   *
   * \param  other The other storage to copy from.
   * \tparam Other The type of the other storage to copy from.
   */
  template <typename Other>
  ripple_host_device auto copy(const Other& other) noexcept -> void {
    static_assert(
      is_storage_accessor_v<Other>,
      "Can only copy from storage accessor types!");
    unrolled_for<num_types>([&](auto i) {
      constexpr size_t type_idx = i;
      using Type                = nth_element_t<type_idx, Ts...>;
      constexpr auto values     = element_components<Type>;
      copy_from_to<type_idx, values, Type>(other, *this);
    });
  }

  /**
   * Returns the number of components in the Ith type being stored. For
   *  non-indexable types this will always return 1, otherwise will return the
   * number of possible components which can be indexed.
   * \tparam I The index of the type to get the number of components for.
   * \return The number of components in the Ith type.
   */
  template <size_t I>
  ripple_host_device constexpr auto components_of() const noexcept -> size_t {
    return nth_element_components<I>;
  }

  /**
   * Gets a reference to the Ith data type.
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A reference to the Ith element.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Ts...>,
    non_vec_element_enable_t<T> = 0>
  ripple_host_device auto get() noexcept -> element_value_t<T>& {
    return *static_cast<element_value_t<T>*>(data_[I]);
  }

  /**
   * Gets a const reference to the Ith data type.
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A const reference to the Ith element.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Ts...>,
    non_vec_element_enable_t<T> = 0>
  ripple_host_device auto get() const noexcept -> const element_value_t<T>& {
    return *static_cast<const element_value_t<T>*>(data_[I]);
  }

  /**
   * Gets a reference to the Jth element of the Ith data type, if the Ith type
   * is indexable.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \tparam T The type of the Ith element.
   * \return A reference to the Jth element in the Ith type.
   */
  template <
    size_t I,
    size_t J,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device auto get() noexcept -> element_value_t<T>& {
    static_assert(
      J < element_components<T>, "Out of range access for element!");
    return static_cast<element_value_t<T>*>(data_[I])[J * stride_];
  }

  /**
   * Gets a const reference to the Jth element of the Ith data type, if the
   * Ith type is indexable.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \tparam T The type of the Ith element.
   * \return A const reference to the Jth element of the Ith type.
   */
  template <
    size_t I,
    size_t J,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device auto get() const noexcept -> const element_value_t<T>& {
    static_assert(
      J < element_components<T>, "Out of range access for element!");
    return static_cast<const element_value_t<T>*>(data_[I])[J * stride_];
  }

  /**
   * Gets a reference to the jth element of the Ith data type, if the Ith type
   * is indexable.
   * \param  j The index of the component in the type to get.
   * \tparam I The index of the type to get the data from.
   * \tparam T The type of the Ith element.
   * \return A reference to the jth element of the Ith type.
   */
  template <
    size_t I,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device auto get(size_t j) noexcept -> element_value_t<T>& {
    return static_cast<element_value_t<T>*>(data_[I])[j * stride_];
  }

  /**
   * Gets a const reference to the jth element of the Ith data type, if the
   * Ith type is indexable. \param  j The index of the component in the type
   * to get. \tparam I The index of the type to get the data from. \tparam T
   * The type of the Ith element. \return A const reference to the jth element
   * of the Ith type.
   */
  template <
    size_t I,
    typename T              = nth_element_t<I, Ts...>,
    vec_element_enable_t<T> = 0>
  ripple_host_device auto
  get(size_t j) const noexcept -> const element_value_t<T>& {
    return static_cast<const element_value_t<T>*>(data_[I])[j * stride_];
  }
};

/*==--- [single type specialization] ---------------------------------------==*/

/**
 * Specialization for strided storage for a single type.
 *
 * The data for this class is stided in that each element of a multi-element
 * type his offset by the stride of the zero dimension, which is the number of
 * elements in the zero dimension, including padding.
 *
 * \tparam Type The underlysing type to create a storage view for.
 */
template <typename Type>
class StridedStorageView<Type>
: public StorageAccessor<StridedStorageView<Type>> {
  // clang-format off
  /** Defiens the type of this storage. */
  using Storage       = StridedStorageView;
  /** Defines the storage element traits for the type. */
  using ElementTraits = storage_element_traits_t<Type>;
  /** Gets the value type of the storage element traits for a type. */
  using ValueType     = typename ElementTraits::Value;
  /** Defines the type of the pointer to the data. */
  using Ptr           = ValueType*;
  /** Defines the type of a const pointer to the data. */
  using ConstPtr      = const ValueType*;
 
  /** LayoutTraits is a friend to allow allocator access. */
  template <typename T, bool B>
  friend struct LayoutTraits;

  /*==--- [constants] ------------------------------------------------------==*/

  // clang-format off
  /** Returns the number of different types. */
  static constexpr size_t num_types          = 1;
  /** Gets the numbber of components for the storage element. */
  static constexpr size_t element_components = ElementTraits::num_elements;
  /** Defines the effective byte size of all elements to store. */
  static constexpr size_t storage_byte_size  = ElementTraits::byte_size;
  /** Gets the number of bytes required for the element type. */
  static constexpr size_t element_bytes      = sizeof(ValueType);
  // clang-format on

  /*==--- [allocator] ------------------------------------------------------==*/

  /**
   * Allocator for the strided storage. This can be used to determine the
   * memory requirement for the storage for a spicifc spatial configuration, as
   * well as to access into the storage space.
   */
  struct Allocator {
   private:
    /**
     * Returns the scaling factor when offsetting in the given dimension.
     * \param  dim The dimension to base the scaling on.
     * \tparam I   The index of the component to get the scaling factor from.
     * \return The scaling factor for offsetting in the given dimension.
     */
    template <typename Dim>
    ripple_host_device static constexpr auto
    offset_scale(Dim&& dim) noexcept -> size_t {
      if constexpr (is_cx_number_v<Dim>) {
        if constexpr (std::decay_t<Dim>::value == DimX::value) {
          return 1;
        } else {
          return element_components;
        }
      } else {
        return dim == 0 ? 1 : element_components;
      }
    }

   public:
    /**
     * Returns the alignment required to allocate the storage.
     */
    static constexpr size_t alignment = ElementTraits::align_size;

    /**
     * Computes the number of bytes required to allocate the the number of
     * specified elements.
     * \param elements The number of elements to allocate.
     * \return The number of bytes required to allocate the number of elements.
     */
    ripple_host_device static constexpr auto
    allocation_size(size_t elements) noexcept -> size_t {
      return storage_byte_size * elements;
    }

    /**
     * Computesthe number of bytes required to allocate a total of Elements
     * of the types defined by Ts. This overload of the function can be used to
     * allocate static memory when the number of elements in the space is known
     * at compile time.
     *
     * \tparam Elements The number of elements to allocate.
     * \return The number of bytes required to allocated the number of elements.
     */
    template <size_t Elements>
    ripple_host_device static constexpr auto
    allocation_size() noexcept -> size_t {
      return storage_byte_size * Elements;
    }

    /**
     * Gets the number of types which are stored strided.
     * \return The number of types which are stored strided.
     */
    static constexpr auto strided_types() noexcept -> size_t {
      return 1;
    }

    /**
     * Gets the number of elements in the Ith type.
     * \tparam I The index of the type.
     * \return The number of elements in the ith type.
     */
    template <size_t I>
    static constexpr auto num_elements() noexcept -> size_t {
      return element_components;
    }

    /**
     * Gets the number of bytes for an elemeent in the Ith type.
     * \tparam I The index of the type to get the element size of.
     * \return The number of bytes for an element in the Ith type.
     */
    template <size_t I>
    static constexpr auto element_byte_size() noexcept -> size_t {
      return element_bytes;
    }

    /**
     * Offsets the storage by the amount specified in the given dimension.
     *
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     * \return A new strided storage offset by the given amount.
     */
    template <typename SpaceImpl, typename Dim>
    ripple_host_device static auto offset(
      const Storage&                  storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) noexcept -> Storage {
      Storage r;
      r.stride_ = storage.stride_;
      r.data_   = storage.data_ + amount * space.step(ripple_forward(dim)) *
                                  offset_scale(ripple_forward(dim));
      return r;
    }

    /**
     * Shifts the storage by the amount specified in the given dimension.
     * \param  storage   The storage to offset.
     * \param  space     The space for which the storage is defined.
     * \param  dim       The dimension to offset in.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Dim       The type of the dimension.
     * \return A new strided storage offset by the given amount.
     */
    template <typename SpaceImpl, typename Dim>
    ripple_host_device static auto shift(
      Storage&                        storage,
      const MultidimSpace<SpaceImpl>& space,
      Dim&&                           dim,
      int                             amount) noexcept -> void {
      storage.data_ = storage.data_ + amount * space.step(ripple_forward(dim)) *
                                        offset_scale(ripple_forward(dim));
    }

    /**
     * Creates the storage, initializing a StridedStorage instance which has
     * its data pointers pointing to the given pointer.
     *
     * \note The memory space should have a size which is that returned by the
     *       `allocation_size()` method, otherwise this may index into undefined
     *       memory.
     *
     * \param  ptr       A pointer to the data to create the storage in.
     * \param  space     The space for which the storage is defined.
     * \param  is        The indices to offset to in the space.
     * \tparam SpaceImpl The implementation of the spatial interface.
     * \tparam Indices   The types of the indices.
     * \return A new strided storage pointing to given pointer.
     */
    template <typename SpaceImpl>
    ripple_host_device static auto
    create(void* ptr, const MultidimSpace<SpaceImpl>& space) noexcept
      -> Storage {
      Storage r;
      r.stride_ = space.size(dimx());
      r.data_   = static_cast<Ptr>(ptr);
      return r;
    }
  };

  /*==--- [members] --------------------------------------------------------==*/

  Ptr      data_   = nullptr; //!< Pointers to the data.
  uint32_t stride_ = 1;       //!< Stride between elements.

 public:
  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor for the strided storage.
   */
  StridedStorageView() = default;

  /**
   * Constructor to set the strided storage from another StorageAccessor with a
   * different layout.
   *
   * \param  other The accessor to set this storage from.
   * \tparam Impl The implementation of the StorageAccessor.
   */
  template <typename Impl>
  ripple_host_device
  StridedStorageView(const StorageAccessor<Impl>& other) noexcept {
    copy(static_cast<const Impl&>(other));
  }

  /**
   * Copy constructor to set the strided storage from the other storage.
   * \param other The other storage to set this one from.
   */
  ripple_host_device
  StridedStorageView(const StridedStorageView& other) noexcept
  : data_{other.data_}, stride_{other.stride_} {}

  /**
   * Move constructor to move the other storage into this one.
   * \param other The other storage to move into this one.
   */
  ripple_host_device StridedStorageView(StridedStorageView&& other) noexcept
  : data_{other.data_}, stride_{other.stride_} {
    other.data_ = nullptr;
  }

  /*==--- [operator overload] ----------------------------------------------==*/

  /**
   * Overload of assignment operator to set the data for the
   * StridedStorageView from  another StorageAccessor. \param  other The
   * accessor to copy the data from. \tparam Impl  The implementation of the
   * StorageAccessor. \return A reference to the created storage view.
   */
  template <typename Impl>
  ripple_host_device auto operator=(const StorageAccessor<Impl>& other) noexcept
    -> StridedStorageView& {
    copy(static_cast<const Impl&>(other));
    return *this;
  }

  /**
   * Overload of assignment operator to set the data for the
   * StridedStorageView from  another StridedStorageView. \param  other The
   * strided storage to copy from. \return A reference to the created storage
   * view.
   */
  ripple_host_device auto
  operator=(const StridedStorageView& other) noexcept -> StridedStorageView& {
    copy(other);
    return *this;
  }

  /**
   * Overload of move assignment operator to move the other strided view into
   * this one.
   *
   *\param  other The strided storage to move from.
   * \return A reference to the created storage view.
   */
  ripple_host_device auto
  operator=(StridedStorageView&& other) noexcept -> StridedStorageView& {
    stride_     = other.stride_;
    data_       = other.data_;
    other.data_ = nullptr;
    return *this;
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets a pointer to the data for the storage.
   * \return A pointer to the data for the storage.
   */
  ripple_host_device auto data() noexcept -> Ptr {
    return &data_;
  }

  /**
   * Gets a const pointer to the data.
   * \return A const pointer to the data for the storage.
   */
  ripple_host_device auto data() const noexcept -> ConstPtr {
    return &data_;
  }

  /**
   * Returns a reference to the data pointers for the storage.
   */
  ripple_host_device auto data_ptrs() noexcept -> std::vector<Ptr> {
    return std::vector<Ptr>{data_};
  }

  /**
   * Copies the data from the other type.
   *
   * \note If the other type is not a StorageAccessor, this will cause a
   * compile time error.
   *
   * \param  other The other storage to copy from.
   * \tparam Other The type of the other storage to copy from.
   */
  template <typename Other>
  ripple_host_device auto copy(const Other& other) noexcept -> void {
    static_assert(
      is_storage_accessor_v<Other>,
      "Can only copy from storage accessor types!");
    copy_from_to<0, element_components, Type>(other, *this);
  }

  /**
   * Returns the number of components in the Ith type being stored. For
   *  non-indexable types this will always return 1, otherwise will return the
   * number of possible components which can be indexed.
   * \tparam I The index of the type to get the number of components for.
   * \return The number of components in the Ith type.
   */
  template <size_t I>
  ripple_host_device constexpr auto components_of() const noexcept -> size_t {
    return element_components;
  }

  /**
   * Gets a reference to the Ith data type.
   * \tparam I The index of the type to get the data from.
   * \return A reference to the Ith element.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Type>,
    non_vec_element_enable_t<T> = 0>
  ripple_host_device auto get() noexcept -> ValueType& {
    static_assert(I == 0, "Element only has one type!");
    return *data_;
  }

  /**
   * Gets a const reference to the Ith data type.
   * \tparam I The index of the type to get the data from.
   * \return A const reference to the Ith element.
   */
  template <
    size_t I,
    typename T                  = nth_element_t<I, Type>,
    non_vec_element_enable_t<T> = 0>
  ripple_host_device auto get() const noexcept -> const ValueType& {
    static_assert(I == 0, "Element only has one type!");
    return *data_;
  }

  /**
   * Gets a reference to the Jth element of the Ith data type, if the Ith type
   * is indexable.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \return A reference to the Jth element in the Ith type.
   */
  template <size_t I, size_t J, vec_element_enable_t<Type> = 0>
  ripple_host_device auto get() noexcept -> ValueType& {
    static_assert(I == 0, "Element only has one type!");
    static_assert(J < element_components, "Out of range access for element!");
    return data_[J * stride_];
  }

  /**
   * Gets a const reference to the Jth element of the Ith data type, if the
   * Ith type is indexable.
   *
   * \tparam I The index of the type to get the data from.
   * \tparam J The index in the type to get.
   * \return A const reference to the Jth element of the Ith type.
   */
  template <size_t I, size_t J, vec_element_enable_t<Type> = 0>
  ripple_host_device auto get() const noexcept -> const ValueType& {
    static_assert(I == 0, "Element only has one type!");
    static_assert(J < element_components, "Out of range access for element!");
    return data_[J * stride_];
  }

  /**
   * Gets a reference to the jth element of the Ith data type, if the Ith type
   * is indexable.
   * \param  j The index of the component in the type to get.
   * \tparam I The index of the type to get the data from.
   * \return A reference to the jth element of the Ith type.
   */
  template <size_t I, vec_element_enable_t<Type> = 0>
  ripple_host_device auto get(size_t j) noexcept -> ValueType& {
    return data_[j * stride_];
  }

  /**
   * Gets a const reference to the jth element of the Ith data type, if the
   * Ith type is indexable. \param  j The index of the component in the type
   * to get. \tparam I The index of the type to get the data from. \tparam T
   * The type of the Ith element. \return A const reference to the jth element
   * of the Ith type.
   */
  template <size_t I, vec_element_enable_t<Type> = 0>
  ripple_host_device auto get(size_t j) const noexcept -> const ValueType& {
    return data_[j * stride_];
  }
};

} // namespace ripple

#endif // RIPPLE_STORAGE_STRIDED_STORAGE_VIEW_HPP

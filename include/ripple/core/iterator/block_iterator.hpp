//==--- ripple/core/container/block_iterator.hpp ----------- -*- C++ -*- ---==//
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
/// \brief This file implements an iterator over a block.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_BLOCK_ITERATOR_HPP
#define RIPPLE_CONTAINER_BLOCK_ITERATOR_HPP

#include "iterator_traits.hpp"
#include <ripple/core/container/array_traits.hpp>
#include <ripple/core/math/math.hpp>
#include <ripple/core/storage/storage_traits.hpp>

namespace ripple {

/// The BlockIterator class defines a iterator over a block, for a given space
/// which defines the region of the block. The iterator only iterates over the
/// internal space of the block, not over the padding, and does not have any
/// knowledge of the where the block is in the global context, or where the
/// iterator is in the block. It is ideal for cases where such information is
/// not required, and operations are relative to the iterator (i.e stencil and
/// operations in shared memory).
///
/// The type T for the iterator can be either a normal type, or type which
/// implements the StridableLayout interface. Regardless, the use is the same,
/// and the iterator operator as if it was a pointer to T.
///
/// \tparam T      The data type which the iterator will access.
/// \tparam Space  The type which defines the iteration space.
template <typename T, typename Space>
class BlockIterator {
 public:
  /// The number of dimensions for the iterator.
  static constexpr size_t dims = space_traits_t<Space>::dimensions;

 private:
  //==--- [traits] ---------------------------------------------------------==//

  // clang-format off
  /// Defines the type of the iterator.
  using self_t          = BlockIterator;
  /// Defines the layout traits for the type t.
  using layout_traits_t = layout_traits_t<T>;
  /// Defines the value type for the iterator.
  using value_t         = typename layout_traits_t::value_t;
  /// Defines the type to return for reference semantics.
  using ref_t           = typename layout_traits_t::iter_ref_t;
  /// Defines the type to return for constant reference semantics.
  using const_ref_t     = typename layout_traits_t::iter_const_ref_t;
  /// Defines the type to return for pointer semantics.
  using ptr_t           = typename layout_traits_t::iter_ptr_t;
  /// Defines the type to return for constant pointer semantics.
  using const_ptr_t     = typename layout_traits_t::iter_const_ptr_t;
  /// Defines the type used when making a copy.
  using copy_t          = typename layout_traits_t::iter_copy_t;
  /// Defines the type of the storage for the iterator.
  using storage_t       = typename layout_traits_t::iter_storage_t;
  /// Defines the type of the class to use to offset the storage.
  using offsetter_t     = typename layout_traits_t::allocator_t;
  /// Defines the type of the space for the iterator.
  using space_t         = Space;
  /// Defines the type of a vector of value_t with matching dimensions.
  using vec_t           = Vector<copy_t, dims, contiguous_owned_t>;
  // clang-format on

  //==--- [constants] ------------------------------------------------------==//

  /// Defines an overload instance for overloading implementations based on the
  /// stridability of the type T.
  static constexpr auto is_stridable_overload_v =
    StridableOverloader<layout_traits_t::is_stridable_layout>{};

  //==--- [deref impl] -----------------------------------------------------==//

  /// Implementation of dereferencing for stridable types. Since the stridable
  /// type stores a pointer like wrapper, a reference to this type is returned.
  ripple_host_device auto deref_impl(stridable_overload_t) -> ref_t {
    return ref_t{_data_ptr};
  }
  /// Implementation of dereferencing for stridable types. Since the stridable
  /// type stores a pointer like wrapper, a constant reference to this type is
  /// returned.
  ripple_host_device auto
  deref_impl(stridable_overload_t) const -> const_ref_t {
    return const_ref_t{_data_ptr};
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
  ripple_host_device auto
  deref_impl(non_stridable_overload_t) const -> const_ref_t {
    return *_data_ptr;
  }

  //==--- [access impl] ----------------------------------------------------==//

  /// Implementation of accessing for stridable types. For a stirdable type, a
  /// pointer like wrapper is stored, rather than a pointer, so the address of
  /// the wrapping type needs to be returned.
  ripple_host_device auto access_impl(stridable_overload_t) -> ptr_t {
    return ptr_t{value_t{_data_ptr}};
  }
  /// Implementation of accessing for stridable types. For a stirdable type, a
  /// pointer like wrapper is stored, rather than a pointer, so the constant
  /// address of the wrapping type needs to be returned.
  ripple_host_device auto
  access_impl(stridable_overload_t) const -> const_ptr_t {
    return const_ptr_t{value_t{_data_ptr}};
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
  ripple_host_device auto
  access_impl(non_stridable_overload_t) const -> const_ptr_t {
    return _data_ptr;
  }

  /// Implementation of unwrapping functionality, to return the type the
  /// iterator iterates over. This overload is for a stridable type.
  ripple_host_device auto unwrap_impl(stridable_overload_t) const -> copy_t {
    return copy_t{_data_ptr};
  }

  /// Implementation of unwrapping functionality, to return the type the
  /// iterator iterates over. This overload is for a non stridable type.
  ripple_host_device auto
  unwrap_impl(non_stridable_overload_t) const -> copy_t {
    return copy_t{*_data_ptr};
  }

  storage_t _data_ptr; //!< A pointer to the data.
  space_t   _space;    //!< The space over which to iterate.

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
  ripple_host_device auto operator-> () -> ptr_t {
    return access_impl(is_stridable_overload_v);
  }

  /// Overload of the access operator to return a pointer, or pointer-like
  /// object for the type T.
  ripple_host_device auto operator-> () const -> const_ptr_t {
    return access_impl(is_stridable_overload_v);
  }

  /// Unwraps the iterator, returning a copy of the data to which the iterator
  /// points.
  ripple_host_device auto unwrap() const -> copy_t {
    return unwrap_impl(is_stridable_overload_v);
  }

  //==--- [offsetting] -----------------------------------------------------==//

  /// Offsets the iterator by \p amount positions in the block in the \p dim
  /// dimension, returning a new iterator offset to the location.
  /// \param  dim    The dimension to offset in
  /// \param  amount The amount to offset by.
  /// \tparam Dim    The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto
  offset(Dim&& dim, int amount = 1) const -> self_t {
    return self_t{offsetter_t::offset(_data_ptr, _space, dim, amount), _space};
  }

  /// Shifts the iterator by \p amount positions in the block in the \p dim
  /// dimension.
  /// \param  dim    The dimension to offset in
  /// \param  amount The amount to offset by.
  /// \tparam Dim    The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto shift(Dim&& dim, int amount = 1) -> void {
    offsetter_t::shift(_data_ptr, _space, dim, amount);
  }

  //==--- [dimensions] -----------------------------------------------------==//

  /// Returns the number of dimensions for the iterator.
  ripple_host_device constexpr auto dimensions() const -> std::size_t {
    return dims;
  }

  //==--- [gradients] ------------------------------------------------------==//

  /// Returns the backward difference between this iterator and the iterator \p
  /// amount places from from this iterator in dimension \p dim.
  ///
  /// \begin{equation}
  ///   \Delta \phi = \phi_{d}(i) - \phi_{d}(i - \textrm{amount})
  /// \end{equation}
  ///
  /// The default is that \p amount is 1, i.e:
  ///
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = it.backward_diff(ripple::dim_x);
  /// auto diff = it.backward_diff(ripple::dim_x, 1);
  /// ~~~
  ///
  /// \param  dim    The dimension to offset in.
  /// \param  amount The amount to offset the iterator by.
  /// \tparam Dim    The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto
  backward_diff(Dim&& dim, unsigned int amount = 1) const -> copy_t {
    return deref_impl(is_stridable_overload_v) -
           *offset(std::forward<Dim>(dim), -static_cast<int>(amount));
  }

  /// Returns the forward difference between this iterator and the iterator \p
  /// amount places from from this iterator in dimension \p dim.
  ///
  /// \begin{equation}
  ///   \Delta \phi = \phi_{d}(i + \textrm{amount}) - \phi_{d}(i)
  /// \end{equation}
  ///
  /// The default is that \p amount is 1, i.e:
  ///
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = it.forward_diff(ripple::dim_x);
  /// auto diff = it.forward_diff(ripple::dim_x, 1);
  /// ~~~
  ///
  /// \param  dim    The dimension to offset in.
  /// \param  amount The amount to offset the iterator by.
  /// \tparam Dim    The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto
  forward_diff(Dim&& dim, unsigned int amount = 1) const -> copy_t {
    return *offset(std::forward<Dim>(dim), amount) -
           deref_impl(is_stridable_overload_v);
  }

  /// Returns the central difference for the cell pointed to by this iterator,
  /// using the iterator \p amount _forward_ and \p amount _backward_ of this
  /// iterator in the \p dim dimension.
  ///
  /// \begin{equation}
  ///   \Delta \phi =
  ///     \phi_{d}(i + \textrm{amount}) - \phi_{d}(i - \textrm{amount})
  /// \end{equation}
  ///
  /// The default is that \p amount is 1, i.e:
  ///
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = it.central_diff(ripple::dim_x);
  /// auto diff = it.central_diff(ripple::dim_x, 1);
  /// ~~~
  ///
  /// \param  dim    The dimension to offset in.
  /// \param  amount The amount to offset the iterator by.
  /// \tparam Dim    The type of the dimension.
  template <typename Dim>
  ripple_host_device constexpr auto
  central_diff(Dim&& dim, unsigned int amount = 1) const -> copy_t {
    return *offset(std::forward<Dim>(dim), amount) -
           *offset(std::forward<Dim>(dim), -static_cast<int>(amount));
  }

  /// Computes the gradient of the data iterated over, as:
  ///
  /// \begin{equation}
  ///   \nabla \phi = (\frac{d}{dx}, .., \frac{d}{dz}) \phi
  /// \end{equation}
  ///
  /// we compute this for a given dimension as:
  ///
  /// \begin{eqution}
  ///   \frac{d \phi}{dx} = \frac{\phi(x + dh) - \phi(x - dh)}{2dh}
  /// \end{equation}
  ///
  /// Returning a vector of N dimensions, with the value in each dimension set
  /// as the gradient in the respective dimension. The gradient uses the central
  /// difference, thus the iterator must be valid on both sides.
  /// \param  dh       The resolution of the grid the iterator iterates over.
  /// \tparam DataType The type of the resolution operator.
  template <typename DataType>
  ripple_host_device constexpr auto grad(DataType dh = 1) const -> vec_t {
    auto result = vec_t();
    unrolled_for<dims>([&](auto d) { result[d] = grad_dim(d, dh); });
    return result;
  }

  /// Computes the gradient of the data iterated over, in a specific dimension.
  ///
  /// \begin{eqution}
  ///   \frac{d \phi}{dx} = \frac{\phi(x + dh) - \phi(x - dh)}{2dh}
  /// \end{equation}
  ///
  /// Returning the type iterated over by the iterator.
  /// \param  dim      The dimension to get the gradient in.
  /// \param  dh       The resolution of the grid the iterator iterates over.
  /// \tparam Dim      The type of the dimension specifier.
  /// \tparam DataType The type of the discretization resolution.
  template <typename Dim, typename DataType>
  ripple_host_device constexpr auto
  grad_dim(Dim&& dim, DataType dh = 1) const -> copy_t {
    // NOTE: Have to do something differenct depending on the data type,
    // because the optimization for 0.5 / dh doesn't work if dh is integral
    // since it goes to zero.
    if constexpr (std::is_integral_v<DataType>) {
      return this->central_diff(dim) / (2 * dh);
    } else {
      return (DataType{0.5} / dh) * this->central_diff(dim);
    }
  }

  //==--- [normal] ---------------------------------------------------------==//

  /// Computes the norm of the data, which is defined as:
  ///
  /// \begin{equation}
  ///   - \frac{\nabla \phi}{|\nabla \phi|}
  /// \end{equation}
  ///
  /// If the data is a vector, this computes the elementwise normal of the
  /// vector.
  ///
  /// \param  dh       The resolution of the grid the iterator iterates over.
  /// \tparam DataType The type of the discretization resolution.
  template <typename DataType>
  ripple_host_device constexpr auto
  norm(DataType dh = DataType(1)) const -> vec_t {
    // NOTE: Here we do not use the grad() function to save some loops.
    auto result = vec_t(0);
    auto mag    = vec_t(0);

    // NOTE: this may need to change to -0.5, as in some of the literature.
    unrolled_for<dims>([&](auto d) {
      // Add the negative sign in now, to avoid an op later ...
      if constexpr (std::is_integral_v<DataType>) {
        result[d] = this->central_diff(d) / (-2 * dh);
      } else {
        result[d] = (DataType{-0.5} / dh) * this->central_diff(d);
      }
      mag += result[d] * result[d];
    });
    result /= math::sqrt(mag);
    return result;
  }

  /// Computes the norm of the data, which is defined as:
  ///
  /// \begin{equation}
  ///   -\frac{\nabla \phi}{|\nabla \phi|}
  /// \end{equation}
  ///
  /// for the case that it is known that $phi$ is a signed distance function and
  /// hence that $|\nabla \phi| = 1$.
  ///
  /// In this case, the computation of the magnitude, and the subsequent
  /// division by it can be avoided which is a significant performance increase.
  ///
  /// \param  dh       The resolution of the grid the iterator iterates over.
  /// \tparam DataType The type of descretization resolution.
  template <typename DataType>
  ripple_host_device constexpr auto norm_sd(DataType dh = 1) const -> vec_t {
    // NOTE: Don't use grad() to save operations for vector types.
    auto result = vec_t();
    unrolled_for<dims>([&](auto d) {
      // Add the negative sign in now, to avoid an op later ...
      if constexpr (std::is_integral_v<DataType>) {
        result[d] = this->central_diff(d) / (-2 * dh);
      } else {
        result[d] = (DataType{-0.5} / dh) * this->central_diff(d);
      }
    });
    return result;
  }

  //==--- [size] -----------------------------------------------------------==//

  /// Returns the total size of the iteration space.
  ripple_host_device constexpr auto size() const -> std::size_t {
    return _space.internal_size();
  }

  /// Returns the size of the iteration space in the given dimension \p dim.
  /// \param  dim The dimension to get the size of.
  /// \tparam Dim The type of the dimension specifier.
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const -> std::size_t {
    return _space.internal_size(std::forward<Dim>(dim));
  }

  /// Returns the amount of padding for the iteration space.
  ripple_host_device constexpr auto padding() const -> std::size_t {
    return _space.padding();
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BOX_ITERATOR_HPP

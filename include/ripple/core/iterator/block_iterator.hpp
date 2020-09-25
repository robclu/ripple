//==--- ripple/core/container/block_iterator.hpp ----------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
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
#include <ripple/core/execution/thread_index.hpp>
#include <cassert>

namespace ripple {

/**
 * The BlockIterator class defines a iterator over a block, for a given space
 * which defines the region of the block. The iterator only iterates over the
 * internal space of the block, not over the padding, and does not have any
 * knowledge of the where the block is in the global context, or where the
 * iterator is in the block. It is ideal for cases where such information is
 * not required, and operations are relative to the iterator (i.e stencil-like
 * operations, operations which required neighbour data, and work on shared
 * memory data).
 *
 * The type T for the iterator can be either a normal type, or type which
 * implements the StridableLayout interface. Regardless, the use is the same,
 * and the iterator operator as if it was a pointer to T.
 *
 * \todo Modify the implementation to take a reference to the space, so that
 *       the space can be stored in shared memory due to the iterator requiring
 *       a significant number of registers in 3D.
 *
 * \tparam T      The data type which the iterator will access.
 * \tparam Space  The type which defines the iteration space.
 */
template <typename T, typename Space>
class BlockIterator {
 public:
  /** The number of dimensions for the iterator. */
  static constexpr size_t dims = space_traits_t<Space>::dimensions;

 private:
  /*==--- [traits] ---------------------------------------------------------==*/

  // clang-format off
  /** Defines the type of the iterator. */
  using self_t          = BlockIterator;
  /**  Defines the layout traits for the type t. */
  using layout_traits_t = layout_traits_t<T>;
  /** Defines the value type for the iterator. */
  using value_t         = typename layout_traits_t::value_t;
  /** Defines the type to return for reference semantics. */
  using ref_t           = typename layout_traits_t::iter_ref_t;
  /** Defines the type to return for constant reference semantics. */
  using const_ref_t     = typename layout_traits_t::iter_const_ref_t;
  /** Defines the type to return for pointer semantics. */
  using ptr_t           = typename layout_traits_t::iter_ptr_t;
  /** Defines the type to return for constant pointer semantics. */
  using const_ptr_t     = typename layout_traits_t::iter_const_ptr_t;
  /** Defines the type used when making a copy. */
  using copy_t          = typename layout_traits_t::iter_copy_t;
  /** Defines the type of the storage for the iterator. */
  using storage_t       = typename layout_traits_t::iter_storage_t;
  /** Defines the type of the class to use to offset the storage. */
  using offsetter_t     = typename layout_traits_t::allocator_t;
  /** Defines the type of the space for the iterator. */
  using space_t         = Space;
  /** Defines the type of a vector of value_t with matching dimensions. */
  using vec_t           = Vector<copy_t, dims, contiguous_owned_t>;
  // clang-format on

  /*==--- [constants] ------------------------------------------------------==*/

  /**
   * Defines an overloaded instance for overloading implementations based on the
   * stridability of the iterated type.
   */
  static constexpr auto is_stridable_overload_v =
    StridableOverloader<layout_traits_t::is_stridable_layout>{};

  /*==--- [deref impl] -----------------------------------------------------==*/

  /**
   * Implementation of dereferencing for stridable types. Since the stridable
   * type stores a pointer like wrapper.
   * \return A reference to the iterated type.
   */
  ripple_host_device auto deref_impl(stridable_overload_t) noexcept -> ref_t {
    return ref_t{_data_ptr};
  }

  /**
   * Implementation of dereferencing for stridable types. Since the stridable
   * type stores a pointer like wrapper.
   * \return A constant reference to the iterated type.
   */
  ripple_host_device auto
  deref_impl(stridable_overload_t) const noexcept -> const_ref_t {
    return const_ref_t{_data_ptr};
  }

  /**
   * Implementation of dereferencing for non stridable types. Since for regular
   * types the iterator stores a pointer to the type, dereferencing is required
   * here.
   * \return A referened to the iterated type.
   */
  ripple_host_device auto
  deref_impl(non_stridable_overload_t) noexcept -> ref_t {
    return *_data_ptr;
  }

  /**
   * Implementation of dereferencing for non stridable types. Since for regular
   * types the iterator stores a pointer to the type, dereferencing is required
   * here.
   * \return A const reference to the iterated type.
   */
  ripple_host_device auto
  deref_impl(non_stridable_overload_t) const noexcept -> const_ref_t {
    return *_data_ptr;
  }

  /*==--- [access impl] ----------------------------------------------------==*/

  /**
   * Implementation of accessing for stridable types.
   *
   * \note For a stridable type, a pointer like wrapper is stored, rather than a
   *       pointer, so the address of the wrapping type needs to be returned.
   *
   * \return A wrapper type over the data with pointer semantics.
   */
  ripple_host_device auto access_impl(stridable_overload_t) noexcept -> ptr_t {
    return ptr_t{value_t{_data_ptr}};
  }

  /**
   * Implementation of accessing for stridable types.
   *
   * \note For a stridable type, a pointer like wrapper is stored, rather than a
   *       pointer, so the constant address of the wrapping type needs to be
   *       returned.
   *
   * \return A const wrapper type over the data, with pointer semantics.
   */
  ripple_host_device auto
  access_impl(stridable_overload_t) const noexcept -> const_ptr_t {
    return const_ptr_t{value_t{_data_ptr}};
  }

  /**
   * Implementation of accessing for non-stridable types.
   *
   * \note For a non-stridable type, a pointer is stored, so this can just
   *       be returned without taking the address.
   *
   * \return A pointer to the iterated type.
   */
  ripple_host_device auto
  access_impl(non_stridable_overload_t) noexcept -> ptr_t {
    return _data_ptr;
  }

  /**
   * Implementation of accessing for non-stridable types.
   *
   * \note For a non-stridable type, a pointer is stored, so this can just be
   *       returned without taking the address.
   *
   * \return A const pointer to the iterated type.
   */
  ripple_host_device auto
  access_impl(non_stridable_overload_t) const noexcept -> const_ptr_t {
    return _data_ptr;
  }

  /*===--- [unwrap impl] ---------------------------------------------------==*/

  /**
   * Implementation of unwrapping functionality. This overload is for stridable
   * types.
   * \return A copy of the iterated type.
   */
  ripple_host_device auto
  unwrap_impl(stridable_overload_t) const noexcept -> copy_t {
    return copy_t{_data_ptr};
  }

  /**
   * Implementation of unwrapping functionality. This overload is for
   * non-stridtable types.
   * \return A copy of the iterated type.
   */
  ripple_host_device auto
  unwrap_impl(non_stridable_overload_t) const noexcept -> copy_t {
    return copy_t{*_data_ptr};
  }

  storage_t _data_ptr; //!< A pointer to the data.
  space_t   _space;    //!< The space over which to iterate.

 public:
  // clang-format off
  /** Defines the type of the raw pointer to the data. */
  using raw_ptr_t       = typename layout_traits_t::raw_ptr_t;
  /** Defines the type of a const raw pointer to the data. */
  using const_raw_ptr_t = typename layout_traits_t::const_raw_ptr_t;
  // clang-format on

  /**
   * Constructor to create the iterator from the storage type and a space over
   * which the iterator can iterate. If the type T is a StridableLayout type,
   * then the storage must be an implementation of the StorageAccessor
   * interface, otherwise (for regular types) the storage must be a pointer to
   * the type.
   * \param data_ptr A pointer (or type which points) to the data.
   * \param space    The space over which the iterator can iterate.
   */
  ripple_host_device BlockIterator(storage_t data_ptr, space_t space) noexcept
  : _data_ptr{data_ptr}, _space{space} {}

  /*==--- [operator overloading] -------------------------------------------==*/

  /**
   * Overload of the dereference operator to access the type T pointed to by
   * the iterator.
   * \return A reference to the type stored in the iterator.
   */
  ripple_host_device auto operator*() noexcept -> ref_t {
    return deref_impl(is_stridable_overload_v);
  }

  /**
   * Overload of the dereference operator to access the type T pointed to by
   * the iterator.
   * \return A const reference to the type T pointer to by the iterator.
   */
  ripple_host_device auto operator*() const noexcept -> const_ref_t {
    return deref_impl(is_stridable_overload_v);
  }

  // clang-format off
  /**
   * Overload of the access operator to access the underlying data.
   * \return A pointer, or pointer-like object for the iterated type.
   */
  ripple_host_device auto operator->() noexcept -> ptr_t {
    return access_impl(is_stridable_overload_v);
  }

  /**
   * Overload of the access operator to access the underlying data.
   * \return A pointer, or pointer-like oject to the iterated type.
   */
  ripple_host_device auto operator->() const noexcept -> const_ptr_t {
    return access_impl(is_stridable_overload_v);
  }
  // clang-format on

  /**
   * Unwraps the iterated type.
   * \return A copy of the data to which the iterator points.
   */
  ripple_host_device auto unwrap() const noexcept -> copy_t {
    return unwrap_impl(is_stridable_overload_v);
  }

  /*==--- [offsetting] -----------------------------------------------------==*/

  /**
   * Offsets the iterator by \p amount positions in the block in the \p dim
   * dimension.
   * \param  dim    The dimension to offset in
   * \param  amount The amount to offset by.
   * \tparam Dim    The type of the dimension specifier.
   * \return A new iterator to the the offset location.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  offset(Dim&& dim, int amount = 1) const noexcept -> self_t {
    return self_t{
      offsetter_t::offset(_data_ptr, _space, std::forward<Dim>(dim), amount),
      _space};
  }

  /**
   * Shifts the iterator by \p amount positions in the block in the \p dim
   * dimension. This modifies the iterator to be at the shifted location.
   * \param  dim    The dimension to offset in
   * \param  amount The amount to offset by.
   * \tparam Dim    The type of the dimension specifier.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  shift(Dim&& dim, int amount = 1) noexcept -> void {
    offsetter_t::shift(_data_ptr, _space, std::forward<Dim>(dim), amount);
  }

  /**
   * Provides access to the underlying data for the iterator.
   * \return A pointer to the underlying data.
   */
  ripple_host_device auto data() noexcept -> raw_ptr_t {
    if constexpr (is_storage_accessor_v<storage_t>) {
      return _data_ptr.data();
    } else {
      return _data_ptr;
    }
  }

  /**
   * Provides const access to the underlying data for the iterator.
   * \return A pointer to the underlying data.
   */
  ripple_host_device auto data() const noexcept -> const_raw_ptr_t {
    if constexpr (is_storage_accessor_v<storage_t>) {
      return _data_ptr.data();
    } else {
      return _data_ptr;
    }
  }

  /**
   * Determines if the iterator is valid in the dimension, that is, its index is
   * less than the size of the global grid.
   * \param  dim The dimension to check validity in.
   * \tparam Dim The type of the dimension specifier.
   * \return true if the iterator is valid.
   */
  template <typename Dim>
  ripple_host_device auto is_valid(Dim&& dim) const noexcept -> bool {
    return ::ripple::global_idx(dim) < size(dim);
  }

  /*==--- [dimensions] -----------------------------------------------------==*/

  /**
   * Provides the number of dimension for the iterator.
   * \return The number of dimensions for the iterator.
   */
  ripple_host_device constexpr auto dimensions() const noexcept -> size_t {
    return dims;
  }

  /*==--- [gradients] ------------------------------------------------------==*/

  /**
   * Computes the backward difference between this iterator and the iterator \p
   * amount places from from this iterator in dimension \p dim.
   *
   * \begin{equation}
   *   \Delta \phi = \phi_{d}(i) - \phi_{d}(i - \textrm{amount})
   * \end{equation}
   *
   * The default is that \p amount is 1, i.e:
   *
   * ~~~{.cpp}
   * // The following is the same:
   * auto diff = it.backward_diff(ripple::dim_x);
   * auto diff = it.backward_diff(ripple::dim_x, 1);
   * ~~~
   *
   * \note If the iterated data is a vector type, the difference is computed
   *       elementwise for each component.
   *
   * \param  dim    The dimension to offset in.
   * \param  amount The amount to offset the iterator by.
   * \tparam Dim    The type of the dimension.
   * \return The backward difference of this iterator and the one behind it in
   *         the given dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  backward_diff(Dim&& dim, unsigned int amount = 1) const noexcept -> copy_t {
    return deref_impl(is_stridable_overload_v) -
           *offset(std::forward<Dim>(dim), -static_cast<int>(amount));
  }

  /**
   * Computes the forward difference between this iterator and the iterator \p
   * amount places from from this iterator in dimension \p dim.
   *
   * \begin{equation}
   *   \Delta \phi = \phi_{d}(i + \textrm{amount}) - \phi_{d}(i)
   * \end{equation}
   *
   * The default is that \p amount is 1, i.e:
   *
   * ~~~{.cpp}
   * // The following is the same:
   * auto diff = it.forward_diff(ripple::dim_x);
   * auto diff = it.forward_diff(ripple::dim_x, 1);
   * ~~~
   *
   * \note If the iterated data is a vector type, the difference is computed
   *       elementwise for each component.
   *
   * \param  dim    The dimension to offset in.
   * \param  amount The amount to offset the iterator by.
   * \tparam Dim    The type of the dimension.
   * \return The forward difference of the iterator ahead of this iterator and
   *         this iterator in the given dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  forward_diff(Dim&& dim, unsigned int amount = 1) const noexcept -> copy_t {
    return *offset(std::forward<Dim>(dim), amount) -
           deref_impl(is_stridable_overload_v);
  }

  /**
   * Computes the central difference for the cell pointed to by this iterator,
   * using the iterator \p amount _forward_ and \p amount _backward_ of this
   * iterator in the \p dim dimension.
   *
   * \begin{equation}
   *   \Delta \phi =
   *     \phi_{d}(i + \textrm{amount}) - \phi_{d}(i - \textrm{amount})
   * \end{equation}
   *
   * The default is that \p amount is 1, i.e:
   *
   * ~~~{.cpp}
   * // The following is the same:
   * auto diff = it.central_diff(ripple::dim_x);
   * auto diff = it.central_diff(ripple::dim_x, 1);
   * ~~~
   *
   * \note If the iterated data is a vector type, the difference is computed
   *       elementwise for each component.
   *
   * \param  dim    The dimension to offset in.
   * \param  amount The amount to offset the iterator by.
   * \tparam Dim    The type of the dimension.
   * \return The difference between the underlying data ahead and behind of
   *         this iterator in the given dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  central_diff(Dim&& dim, unsigned int amount = 1) const noexcept -> copy_t {
    return *offset(std::forward<Dim>(dim), amount) -
           *offset(std::forward<Dim>(dim), -static_cast<int>(amount));
  }

  /**
   * Computes the second derivative with respect to the given dimension.
   *
   * \begin{equation}
   *   \phi_{dd} =
   *      \phi_{d + \textrm{amount}) -
   *      2 \phi_{d} +
   *      \phi_{d - \textrm{amount})
   * \end{equation}
   *
   * \note This does not do the divison by $h^2$, as it assumes that it is 1.
   *
   * The default is that \p amount is 1, i.e:
   *
   * ~~~{.cpp}
   * // The following is the same:
   * auto diff = it.second_diff(ripple::dim_x);
   * auto diff = it.second_diff(ripple::dim_x, 1);
   * ~~~
   *
   * \note If the iterated data is a vector type, the difference is computed
   *       elementwise for each component.
   *
   * \param  dim    The dimension to offset in.
   * \param  amount The amount to offset the iterator by.
   * \tparam Dim    The type of the dimension.
   * \return The second difference in the given dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto
  second_diff(Dim&& dim, unsigned int amount = 1) const noexcept -> copy_t {
    return *offset(std::forward<Dim>(dim), amount) +
           *offset(std::forward<Dim>(dim), -static_cast<int>(amount)) -
           (T(2) * this->operator*());
  }

  /**
   * Computes the second partialderivative with respect to the given dimensions:
   *
   * \begin{equation}
   *   \phi_{dd} =
   *    \frac{
   *      \phi_{d1 + \textrm{a}, d2 + \textrm{a}) -
   *      \phi_{d1 + \textrm{a}, d2 - \textrm{a}) -
   *      \phi_{d1 - \textrm{a}, d2 + \textrm{a}) +
   *      \phi_{d1 - \textrm{a}, d2 - \textrm{a})
   *    }{4}
   * \end{equation}
   *
   * \note This does not do the divison by $h^2$, as it assumes that it is 1.
   *
   * The default is that \p amount is 1, i.e:
   *
   * ~~~{.cpp}
   * // The following is the same:
   * auto diff = it.second_diff(ripple::dim_x);
   * auto diff = it.second_diff(ripple::dim_x, 1);
   * ~~~
   *
   * \note If the iterated data is a vector type, the difference is computed
   *       elementwise for each component.
   *
   * \param  dim    The dimension to offset in.
   * \param  amount The amount to offset the iterator by.
   * \tparam Dim    The type of the dimension.
   * \return The second difference in the given dimension.
   */
  template <typename Dim1, typename Dim2>
  ripple_host_device constexpr auto
  second_partial_diff(Dim1&& dim1, Dim2&& dim2, unsigned int amount = 1) const
    noexcept -> copy_t {
    const auto scale   = 0.25;
    const int  namount = -static_cast<int>(amount);
    const auto next    = offset(std::forward<Dim1>(dim1), amount);
    const auto prev    = offset(std::forward<Dim1>(dim1), namount);

    return scale * ((*next.offset(std::forward<Dim2>(dim2), amount)) -
                    (*next.offset(std::forward<Dim2>(dim2), namount)) -
                    (*prev.offset(std::forward<Dim2>(dim2), amount)) +
                    (*prev.offset(std::forward<Dim2>(dim2), namount)));
  }

  /**
   * Computes the gradient of the data iterated over, as:
   *
   * \begin{equation}
   *   \nabla \phi = \left[ \frac{d}{dx}, .., \frac{d}{dn} \right] \phi
   * \end{equation}
   *
   * For a given dimension, the compuatation is:
   *
   * \begin{eqution}
   *   \frac{d \phi}{dx} = \frac{\phi(x + dh) - \phi(x - dh)}{2 dh}
   * \end{equation}
   *
   * \note The gradient computation for each deimension uses the central
   *       difference, thus the iterator must be valid on both sides in all
   *       dimensions, otherwise the resulting behaviour is undefined.
   *
   * \note If the iterated data is a vector type, the gradient is computed
   *       elementwise for each component.
   *
   * \see grad_dim
   *
   * \param  dh       The resolution of the grid the iterator iterates over.
   * \tparam DataType The type of the resolution operator.
   * \return A vector of N dimensions, with the value in each dimension set to
   *         the gradient in the given dimension.
   */
  template <typename DataType>
  ripple_host_device constexpr auto
  grad(DataType dh = 1) const noexcept -> vec_t {
    auto result = vec_t();
    unrolled_for<dims>([&](auto d) { result[d] = grad_dim(d, dh); });
    return result;
  }

  /**
   * Computes the gradient of the iterated data in a specific dimension as:
   *
   * \begin{eqution}
   *   \frac{d \phi}{dx} = \frac{\phi(x + dh) - \phi(x - dh)}{2dh}
   * \end{equation}
   *
   * \note The gradient computation uses the central difference, so the data
   *       on each side of the iterator must be valid, otherwise the resulting
   *       behaviour is undefined.
   *
   * \note If the iterated data is a vector type, the gradient is computed
   *       elementwise for each component.
   *
   * \param  dim      The dimension to get the gradient in.
   * \param  dh       The resolution of the grid the iterator iterates over.
   * \tparam Dim      The type of the dimension specifier.
   * \tparam DataType The type of the discretization resolution.
   * \return The gradient of the iterated data in the given dimension.
   */
  template <typename Dim, typename DataType>
  ripple_host_device constexpr auto
  grad_dim(Dim&& dim, DataType dh = 1) const noexcept -> copy_t {
    // NOTE: Have to do something different depending on the data type
    // because the optimization for 0.5 / dh doesn't work if dh is integral
    // since it goes to zero.
    //
    // The lack of optimization in integer branch means that a division will
    // happen for each element if the iterated types has multiple elements,
    // where as for the non-integer branch, the pre-division means that the
    // divisions are turned into multiplications, which is a lot faster,
    // especially on the device.
    if constexpr (std::is_integral_v<DataType>) {
      return this->central_diff(dim) / (2 * dh);
    } else {
      return (DataType{0.5} / dh) * this->central_diff(dim);
    }
  }

  /*==--- [normal] ---------------------------------------------------------==*/

  /**
   * Computes the norm of the data, which is defined as:
   *
   * \begin{equation}
   *   - \frac{\nabla \phi}{|\nabla \phi|}
   * \end{equation}
   *
   * \note If the iterated data is a vector type the computation is performed
   *       elementwise for each component of the vector type.
   *
   * \note This will assert in debug if the division is too close to zero.
   *
   * \param  dh       The resolution of the grid the iterator iterates over.
   * \tparam DataType The type of the discretization resolution.
   * \return The norm of the iterated data.
   */
  template <typename DataType>
  ripple_host_device constexpr auto
  norm(DataType dh = DataType(1)) const noexcept -> vec_t {
    auto result = vec_t{};
    auto mag    = DataType{1e-15};

    // NOTE: Here we do not use the grad() function to save some loops.
    // If grad was used, then an extra multiplication would be required per
    // element for vector types.
    unrolled_for<dims>([&](auto d) {
      // Add the negative sign in now, to avoid an op later ...
      if constexpr (std::is_integral_v<DataType>) {
        result[d] = this->central_diff(d) / (-2 * dh);
      } else {
        result[d] = (DataType{-0.5} / dh) * this->central_diff(d);
      }
      mag += result[d] * result[d];
    });
    // assert(std::abs(std::sqrt(mag)) < 1e-22 && "Division by zero in norm!");
    result /= std::sqrt(mag);
    return result;
  }

  /**
   * Computes the norm of the data, which is defined as:
   *
   * \begin{equation}
   *   -\frac{\nabla \phi}{|\nabla \phi|}
   * \end{equation}
   *
   * for the case that it is known that $phi$ is a signed distance function and
   * hence that $|\nabla \phi| = 1$.
   *
   * In this case, the computation of the magnitude, and the subsequent
   * division by its square root can be avoided, which is a significant
   * performance improvement.
   *
   * \note If the iterated data is a vector type the computation is performed
   *       elementwise for each component of the vector type.
   *
   * \param  dh       The resolution of the grid the iterator iterates over.
   * \tparam DataType The type of descretization resolution.
   * \return The norm of the iterated data.
   */
  template <typename DataType>
  ripple_host_device constexpr auto
  norm_sd(DataType dh = 1) const noexcept -> vec_t {
    // NOTE: Here we do not use the grad() function to save some loops.
    // If grad was used, then an extra multiplication would be required per
    // element for vector types.
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

  /*==--- [curvature] ------------------------------------------------------==*/

  /**
   * Computes the curvature for the iterator, which is defined as:
   *
   * \begin{equation}
   *    \kappa = \nabla \cdot \eft( \frac{\nabla \phi}{|\nabla \phi|} \right)
   * \end{equation}
   *
   * \param  dh       The resolution of the iteration domain.
   * \tparam DataType The type of the data for the resolution.
   * \return A value of the curvature.
   */
  template <typename DataType>
  ripple_host_device constexpr auto
  curvature(DataType dh = 1) const noexcept -> DataType {
    if constexpr (dims == 2) {
      return curvature_2d(dh);
    } else if (dims == 3) {
      return curvature_3d(dh);
    } else {
      return DataType{1};
    }
  }

  /**
   * Computes the curvature for the iterator, which is defined as:
   *
   * \begin{equation}
   *    \kappa = \nabla \cdot \eft( \frac{\nabla \phi}{|\nabla \phi|} \right)
   * \end{equation}
   *
   * \param  dh       The resolution of the iteration domain.
   * \tparam DataType The type of the data for the resolution.
   * \return A value of the curvature.
   */
  template <typename DataType>
  ripple_host_device constexpr auto
  curvature_2d(DataType dh = 1) const noexcept -> DataType {
    const auto px      = grad_dim(dim_x, dh);
    const auto py      = grad_dim(dim_y, dh);
    const auto px2     = px * px;
    const auto py2     = py * py;
    const auto px2_py2 = px2 + py2;
    const auto dh2     = DataType(1) / (dh * dh);
    const auto pxx     = second_diff(dim_x);
    const auto pyy     = second_diff(dim_y);
    const auto pxy     = second_partial_diff(dim_x, dim_y);

    return dh2 * (pxx * py2 - DataType{2} * py * px * pxy + pyy * px2) /
           math::sqrt(px2_py2 * px2_py2 * px2_py2);
  }

  /**
   * Computes the curvature for the iterator, which is defined as:
   *
   * \begin{equation}
   *    \kappa = \nabla \cdot \eft( \frac{\nabla \phi}{|\nabla \phi|} \right)
   * \end{equation}
   *
   * \param  dh       The resolution of the iteration domain.
   * \tparam DataType The type of the data for the resolution.
   * \return A value of the curvature.
   */
  template <typename DataType>
  ripple_host_device constexpr auto
  curvature_3d(DataType dh = 1) const noexcept -> DataType {
    const auto px      = grad_dim(dim_x, dh);
    const auto py      = grad_dim(dim_y, dh);
    const auto px2     = px * px;
    const auto py2     = py * py;
    const auto px2_py2 = px2 + py2;
    const auto dh2     = DataType(1) / (dh * dh);
    const auto pxx     = second_diff(dim_x);
    const auto pyy     = second_diff(dim_y);
    const auto pxy     = second_partial_diff(dim_x, dim_y);

    return dh2 * (pxx * py2 - DataType{2} * py * px * pxy + pyy * px2) /
           math::sqrt(px2_py2 * px2_py2 * px2_py2);
  }

  /*==--- [size] -----------------------------------------------------------==*/

  /**
   * Gets the total size of the iteration space.
   * \note The size *does not* include padding elements.
   * \return The total number of elements in the iteration space.
   */
  ripple_host_device constexpr auto size() const noexcept -> size_t {
    return _space.internal_size();
  }

  /**
   * Getss the size of the iteration space in the given dimension \p dim.
   * \note The size *does not* include padding elements.
   * \param  dim The dimension to get the size of.
   * \tparam Dim The type of the dimension specifier.
   * \return The number of elements in the iteration space for the given
   *         dimension.
   */
  template <typename Dim>
  ripple_host_device constexpr auto size(Dim&& dim) const noexcept -> size_t {
    return _space.internal_size(std::forward<Dim>(dim));
  }

  /**
   * Gets the amount of padding for the iteration space.
   *
   * \note All dimensions have this amount of padding on each side of the
   *       dimension.
   *
   * \return The amount of padding for a single side of a dimension in the
   *         iteration space.
   */
  ripple_host_device constexpr auto padding() const noexcept -> size_t {
    return _space.padding();
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_BOX_ITERATOR_HPP

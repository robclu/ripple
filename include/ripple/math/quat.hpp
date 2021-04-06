/**=--- ripple/math/quat.hpp ------------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  quat.hpp
 * \brief This file defines an implementation for a quaternion.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_MATH_QUAT_HPP
#define RIPPLE_MATH_QUAT_HPP

#include <ripple/container/array.hpp>
#include <ripple/container/tuple.hpp>
#include <ripple/storage/polymorphic_layout.hpp>
#include <ripple/storage/storage_descriptor.hpp>
#include <ripple/storage/storage_traits.hpp>
#include <ripple/storage/struct_accessor.hpp>
#include <ripple/utility/portability.hpp>

namespace ripple {

/**
 * The Quat class implements a quaternion class with polymorphic data layout.
 *
 * The data for the elements is allocated according to the layout, and can be
 * contiguous, owned, or strided.
 *
 * \tparam T      The type of the elements in the quaternion
 * \tparam Layout The type of the storage layout for the quaternion
 */
template <typename T, typename Layout = ContiguousOwned>
struct Quat : public PolymorphicLayout<Quat<T, Layout>> {
 private:
  /*==--- [constants] ------------------------------------------------------==*/

  /**
   * Defines the number of elements in the quaternion, one scaler, one vector.
   */
  static constexpr auto elements = 4;

  //==--- [aliases] --------------------------------------------------------==//

  // clang-format off
  /** Defines the type of the descriptor for the storage. */
  using Descriptor = StorageDescriptor<Layout, Vector<T, elements>>;
  /** Defines the storage type for the array. */
  using Storage    = typename Descriptor::Storage;
  /** Defines the value type of the data in the vector. */
  using Value      = std::decay_t<T>;
  /** Alias for the accessor for the w component. */
  using WAccessor  = StructAccessor<Value, Storage, 0>;
  /** Alias for the accessor for the x component.*/
  using XAccessor  = StructAccessor<Value, Storage, 1>;
  /** Alias for the accessor for the y component.*/
  using YAccessor  = StructAccessor<Value, Storage, 2>;
  /** Alias for the accessor for the z component.*/
  using ZAccessor  = StructAccessor<Value, Storage, 3>;
  // clang-format on
  // clang-format on

  /**
   * Declares quaternions with other storage layouts as friends for
   * construction.
   * \tparam OtherType   The type of the data for the other quaternion.
   * \tparam OtherLayout The layout of the other quaternion.
   */
  template <typename OtherType, typename OtherLayout>
  friend struct Quat;

  /**
   * LayoutTraits is a friend so that it can see the descriptor.
   * \tparam Layable     If the type can be re laid out.
   * \tparam IsStridable If the type is stridable.
   */
  template <typename Layable, bool IsStridable>
  friend struct LayoutTraits;

 public:
  union {
    Storage   storage; //!< The storage for the quaternion.
    WAccessor w;       //!< Scalar component of the quaternion.
    XAccessor x;       //!< X vector component of the quaternion.
    YAccessor y;       //!< Y vector component of the quaternion.
    ZAccessor z;       //!< Z vector component of the quaternion.
  };

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Default constructor for the quaternion, value of initialization for
   * elements in undefined.
   */
  ripple_host_device constexpr Quat() noexcept {}

  /**
   * Sets all elements of the quaternion to the given value.
   * \param val The value to set all elements to.
   */
  ripple_host_device constexpr Quat(T val) noexcept {
    unrolled_for<elements>([&](auto i) { storage.template get<0, i>() = val; });
  }

  /**
   * Constructor to create the quaternion from a list of values. The first
   * value is the scalar component, the rest are the vector component.
   *
   * \note The order of initialization is {scalar, vx, vy, vz}, and elements
   *       which are not specified are set to zero, i.e if 1, 1 are specified
   *       then the quaternion is initialized to `{1, [1 0 0]}`.
   *
   * \note This overload is only enabled when the number of elements in the
   *       variadic parameter is two or more.
   *
   * \note The types of the values must be convertible to T.
   *
   * \param  values The values to set the elements to.
   * \tparam Values The types of the values for setting.
   */
  template <typename... Values, variadic_ge_enable_t<2, Values...> = 0>
  ripple_host_device constexpr Quat(Values&&... values) noexcept {
    const auto       v         = Tuple<Values...>{values...};
    constexpr size_t arg_count = sizeof...(Values);
    constexpr size_t extra     = elements - arg_count;
    unrolled_for<arg_count>(
      [&](auto i) { storage.template get<0, i>() = get<i>(v); });

    unrolled_for<extra>([&](auto i) {
      constexpr size_t idx           = i + arg_count;
      storage.template get<0, idx>() = Value{0};
    });
  }

  /**
   * Constructor to set the quaternion from the other storage.
   * \param other The other storage to use to set the quaternion.
   */
  ripple_host_device constexpr Quat(Storage storage) noexcept
  : storage{storage} {}

  /**
   * Copy constructor to set the quaternion from another quaternion.
   * \param other The other quaternion to use to initialize this one.
   */
  ripple_host_device constexpr Quat(const Quat& other) noexcept
  : storage{other.storage} {}

  /**
   * Move constructor to set the quaternion from another quaternion.
   * \param other The other quaternion to use to initialize this one.
   */
  ripple_host_device constexpr Quat(Quat&& other) noexcept
  : storage{ripple_move(other.storage)} {}

  /**
   * Copy constructor to set the quaternion from another quaternion with
   * different storage layout.
   * \param  other       The other quaternion to use to initialize this one.
   * \tparam OtherLayout The layout of the other storage.
   */
  template <typename OtherLayout>
  ripple_host_device constexpr Quat(const Quat<T, OtherLayout>& other) noexcept
  : storage{other.storage} {}

  /**
   * Move constructor to set the quaternion from another quaternion with a
   * different storage layout.
   * \param  other       The other quaternion to use to initialize this one.
   * \tparam OtherLayout The layout of the other storage.
   */
  template <typename OtherLayout>
  ripple_host_device constexpr Quat(Quat<T, OtherLayout>&& other) noexcept
  : storage{ripple_move(other.storage)} {}

  /*==--- [operator overloads] ---------------------------------------------==*/

  /**
   * Overload of copy assignment overload to copy the elements from another
   * quaternion to this matrix.
   * \param  other The other quaternion to copy from.
   * \return A references to the modified quaternion.
   */
  ripple_host_device auto operator=(const Quat& other) noexcept -> Quat& {
    storage = other.storage;
    return *this;
  }

  /**
   * Overload of move assignment overload to move the elements from another
   * quaternion to this quaternion.
   * \param  other The other quaternion to move.
   * \return A reference to the modified quaternion.
   */
  ripple_host_device auto operator=(Quat&& other) noexcept -> Quat& {
    storage = ripple_move(other.storage);
    return *this;
  }

  /**
   * Overload of copy assignment overload to copy the elements from another
   * quaternion with a different storage layout to this quaternion.
   * \param  other       The other quaternion to copy from.
   * \tparam OtherLayout The layout of the other quaternion.
   * \return A reference to the modified quaternion.
   */
  template <typename OtherLayout>
  ripple_host_device auto
  operator=(const Quat<T, OtherLayout>& other) noexcept -> Quat& {
    unrolled_for<elements>([&](auto i) {
      storage.template get<0, i>() = other.storage.template get<0, i>();
    });
    return *this;
  }

  /**
   * Overload of move assignment overload to copy the elements from another
   * quaternion to this quaternion.
   * \param  other       The other quaternion to move.
   * \tparam OtherLayout The layout of the other quaternion.
   * \return A reference to the modified quaternion.
   */
  template <typename OtherLayout>
  ripple_host_device auto
  operator=(Quat<T, OtherLayout>&& other) noexcept -> Quat& {
    storage = ripple_move(other.storage);
    return *this;
  }

  /**
   * Overload of access operator to get the element at the given index. The
   * access pattern is [scalar, vx, vy, vz].
   * \param i The index of the element to get.
   * \return A reference to the element.
   */
  ripple_host_device auto operator[](size_t i) noexcept -> Value& {
    return storage.template get<0>(i);
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets the scalar part of the quaternion.
   * \return A reference to the scalar part of the quaternion.
   */
  ripple_host_device auto scalar() noexcept -> Value& {
    return storage.template get<0, 0>();
  }

  /**
   * Gets the scalar part of the quaternion.
   * \return A const reference to the scalar part of the quaternion.
   */
  ripple_host_device auto scalar() const noexcept -> const Value& {
    return storage.template get<0, 0>();
  }

  /**
   * Gets the Ith vector component from the quaternion.
   * \tparam I The index of the vector component to get.
   */
  template <size_t I>
  ripple_host_device auto vec_component() noexcept -> Value& {
    return storage.template get<0, I + 1>();
  }

  /**
   * Gets the Ith vector component from the quaternion.
   * \tparam I The index of the vector component to get.
   */
  template <size_t I>
  ripple_host_device auto vec_component() const noexcept -> const Value& {
    return storage.template get<0, I + 1>();
  }

  /**
   * Gets the Ith vector component from the quaternion.
   * \tparam i The index of the vector component to get.
   */
  ripple_host_device auto vec_component(size_t i) noexcept -> Value& {
    return storage.template get<0>(i + 1);
  }

  /**
   * Gets the Ith vector component from the quaternion.
   * \tparam I The index of the vector component to get.
   */
  ripple_host_device auto
  vec_component(size_t i) const noexcept -> const Value& {
    return storage.template get<0>(i + 1);
  }

  /**
   * Gets the length squared of the quaternion.
   * \return The squared length of the quaternion.
   */
  ripple_host_device auto length_squared() const noexcept -> Value {
    return w * w + x * x + y * y + z * z;
  }

  /**
   * Gets the length of the quaternion.
   * \return The length of the quaternion.
   */
  ripple_host_device auto length() const noexcept -> Value {
    return std::sqrt(length_squared());
  }

  /**
   * Normalizes the quaternion.
   */
  ripple_host_device auto normalize() noexcept -> void {
    const auto scale = Value{1} / length();
    w *= scale;
    x *= scale;
    y *= scale;
    z *= scale;
  }

  /**
   * Inverts the quaternion.
   */
  ripple_host_device auto invert() noexcept -> void {
    const auto scale = Value{1} / length_squared();
    w *= scale;
    x *= -scale;
    y *= -scale;
    z *= -scale;
  }

  /**
   * Gets the inverse of the quaternion.
   * \return A new quaternion which is the inverse of this quaternion.
   */
  ripple_host_device auto inverse() const noexcept -> Quat<T, ContiguousOwned> {
    const auto scale = Value{-1} / length_squared();
    return Quat<T, ContiguousOwned>{
      w * -scale, x * scale, y * scale, z * scale};
  }

  /**
   * Multiplies this quaternion with q, returning a new quaternion.
   * \param q The other quaterion to right multiply with.
   */
  template <typename U, typename L>
  ripple_host_device auto
  operator*(const Quat<U, L>& q) const noexcept -> Quat<T, ContiguousOwned> {
    // clang-format off
    return Quat<T, ContiguousOwned>{
      w * q.w - x * q.x - y * q.y - z * q.z,
      w * q.x + x * q.w + y * q.z - z * q.y,
      w * q.y - x * q.z + y * q.w + z * q.x,
      w * q.z + x * q.y - y * q.x + z * q.w};
    // clang-format on
  }
};

/**
 * Gets an ortogonal 2x2 matrix representing the rotation of the quaternion.
 * \param q The  quaternion to get the rotation matrix for.
 * \return A matrix representing the rotation.
 */
template <typename T, typename L>
ripple_host_device auto
to_mat2x2(const Quat<T, L>& q) noexcept -> Mat<T, 2, 2> {
  const auto xy = q.x * q.y;
  const auto wz = q.w * q.z;
  // clang-format off
  return Mat<T, 2, 2>{
    q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
    T{2} * (xy - wz), 
    T{2} * (xy + wz),
    q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z};
  // clang-format on
}

/**
 * Rotates the given vector using the quaternion which defines the rotation
 * plane and angle.
 *
 * With the quaternion defined as q = $(w, \textbf{r})$, with scalar part w and
 * vector part r the the rotated vector is $v^' = v + 2r x (r x v + w * v)$
 * where $x$ is the cross product.
 *
 * \note This overload is for a 3 dimensional vector.
 *
 * \param  q  The quaternion which defines the rotation.
 * \param  v  The vector to rotate.
 * \tparam T  The type of the quaternion data.
 * \tparam LQ The layout of the quaternion.
 * \tparam U  The data type for the vector.
 * \tparam LV The layout of the vector.
 * \return The rotated vector.
 */
template <typename T, typename LQ, typename U, typename LV>
ripple_host_device auto
rotate(const Quat<T, LQ>& q, const Vec3d<U, LV>& v) noexcept
  -> Vec3d<std::decay_t<U>, ContiguousOwned> {
  using TT = std::decay_t<U>;
  // clang-format off
  /* First compute the temp cross product and addition.
     t = r x v + w * v */
  const auto t = Vec3d<TT, ContiguousOwned>{
    q.w * v.x + q.y * v.z - q.z * v.y,
    q.w * v.y + q.z * v.x - q.x * v.z,
    q.w * v.z + q.x * v.y - q.y * v.x
  };
  /* Note: We expand out the multiplication here, so that we don't store any
     temporary results which may bloat the register usage. */
  return Vec3d<TT, ContiguousOwned>{
    v.x + U{2} * (q.y * t.z - q.z * t.y),
    v.y + U{2} * (q.z * t.x - q.x * t.z),
    v.z + U{2} * (q.x * t.y - q.y * t.x)
  };
  // clang-format on
}

/**
 * Rotates the given vector using the quaternion which defines the rotation
 * plane and angle.
 *
 * With the quaternion defined as q = $(w, \textbf{r})$, with scalar part w and
 * vector part r the the rotated vector is $v^' = v + 2r x (r x v + w * v)$
 * where $x$ is the cross product.
 *
 * \note This overload is for a 2 dimensional vector.
 *
 * \param  q  The quaternion which defines the rotation.
 * \param  v  The vector to rotate.
 * \tparam T  The type of the quaternion data.
 * \tparam LQ The layout of the quaternion.
 * \tparam U  The data type for the vector.
 * \tparam LV The layout of the vector.
 * \return The rotated vector.
 */
template <typename T, typename LQ, typename U, typename LV>
ripple_host_device auto
rotate(const Quat<T, LQ>& q, const Vec2d<U, LV>& v) noexcept
  -> Vec2d<std::decay_t<U>, ContiguousOwned> {
  using TT = std::decay_t<U>;
  // clang-format off
  /* First compute the temp cross product and addition.
     t = r x v + w * v */
  const TT x = q.w * v.x - q.z * v.y;
  const TT y = q.w * v.y + q.z * v.x;
  const TT z = q.x * v.y - q.y * v.x;

  /* Note: We expand out the multiplication here, so that we don't store any
     temporary results which may bloat the register usage. */
  return Vec2d<TT, ContiguousOwned>{
    v.x + U{2} * (q.y * z - q.z * y),
    v.y + U{2} * (q.z * x - q.x * z)
  };
  // clang-format on
}

/**
 * Creates a quaternion which represents the rotation from v1 to v2.
 *
 * \note This requires that v1 and v2 be normalized.
 *
 * \note This is a specialization for 3d vectors.
 *
 * \param  v1 The vector to rotate from.
 * \param  v2 The vector to rotate to.
 * \tparam T  The data type for the first vector.
 * \tparam U  The data type for the second vector.
 * \tparam L1 The layout of the first vector.
 * \tparam L2 The layout of the second vector.
 * \return A quaternion which represents the rotation between the vectors.
 */
template <typename T, typename U, typename L1, typename L2>
ripple_host_device auto
create_quat(const Vec3d<T, L1>& v1, const Vec3d<U, L2>& v2) noexcept
  -> Quat<std::decay_t<T>> {
  using TT         = std::decay_t<T>;
  constexpr TT tol = 0.999999999999999999;
  const TT     t   = math::dot(v1, v2);
  Quat<TT>     q{0, 0, 0, 0};

  if (t > tol) {
    q.w = 1;
  } else if (t < -tol) {
    // Here we have to check if the cross with the x/y unit vector is valid,
    // and then define the result in terms of that.
    // We would want to set the quat from the axis (the vector we define
    // here),
    // and the angle (180), but sin and cos terms reduce to 1 and 0, so we
    // don't need to do that.
    const TT scale = TT{1} / v1.length();
    if (std::sqrt(v1.z * v1.z + v1.y * v1.y) >= TT{1} - tol) {
      q.y = -v1.z * scale;
      q.z = v1.y * scale;
    } else {
      q.x = v1.z * scale;
      q.z = -v1.x * scale;
    }
  } else {
    // clang-format off
    q.w = std::sqrt(v1.length_squared() * v2.length_squared()) + t;
    q.x = v1.y * v2.z - v1.z *v2.y;
    q.y = v1.z * v2.x - v1.x *v2.z;
    q.z = v1.x * v2.y - v1.y *v2.x;
    q.normalize();
    // clang-format on
  }
  return q;
}

/**
 * Creates a quaternion which represents the rotation from v1 to v2.
 *
 * \note This requires that v1 and v2 be normalized.
 *
 * \note This is a specialization for 2d vectors.
 *
 * \param  v1 The vector to rotate from.
 * \param  v2 The vector to rotate to.
 * \tparam T  The data type for the first vector.
 * \tparam U  The data type for the second vector.
 * \tparam L1 The layout of the first vector.
 * \tparam L2 The layout of the second vector.
 * \return A quaternion which represents the rotation between the vectors.
 */
template <typename T, typename U, typename L1, typename L2>
ripple_host_device auto
create_quat(const Vec2d<T, L1>& v1, const Vec2d<U, L2>& v2) noexcept
  -> Quat<std::decay_t<T>> {
  using TT         = std::decay_t<T>;
  constexpr TT tol = 0.999999999999999999;
  const TT     t   = math::dot(v1, v2);
  Quat<TT>     q{0, 0, 0, 0};

  if (t > tol) {
    q.w = 1;
  } else if (t < -tol) {
    const auto scale = TT{1} / v1.length();
    if (v1.y >= TT{1} - tol) {
      q.z = v1.y * scale;
    } else {
      q.z = -v1.x * scale;
    }
  } else {
    // clang-format off
    q.w = std::sqrt(v1.length_squared() * v2.length_squared()) + t;
    q.z = v1.x * v2.y - v1.y * v2.x;
    q.normalize();
  }
  return q;
}

} // namespace ripple

#endif // namespace RIPPLE_MATH_MAT_HPP
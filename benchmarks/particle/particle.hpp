//==--- ripple/benchmarks/particle.hpp --------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//

/**=--- ripple/benchmarks/particle.hpp --------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  particle.hpp
 * \brief This file defines an example particle class for bencmarks.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_BENCHMARK_PARTICLE_HPP
#define RIPPLE_BENCHMARK_PARTICLE_HPP

#include <ripple/storage/storage_descriptor.hpp>
#include <ripple/storage/storage_traits.hpp>

using Real = double;

// This is a test class for a particle with a number of dimensions.
template <typename T, typename Dims, typename Layout = ripple::StridedView>
struct Particle : ripple::PolymorphicLayout<Particle<T, Dims, Layout>> {
  /** Dimensions for the particle. */
  static constexpr size_t dims = Dims::value;

  /** Descriptor, one vector each for position and velocity. */
  using Descriptor = ripple::
    StorageDescriptor<Layout, ripple::Vector<T, dims>, ripple::Vector<T, dims>>;

  /** Get the storage type from the descriptor. */
  using Storage = typename Descriptor::Storage;

  /**
   * Constructor from storage, which is required.
   * \param s The storage to create the particle from.
   */
  ripple_host_device Particle(Storage s) noexcept : storage_{s} {}

  /**
   * Constructor from another particle.
   * \param p The other particle to set this one from.
   */
  ripple_host_device Particle(const Particle& p) noexcept
  : storage_{p.storage_} {}

  /**
   * Constructor for particle of a different layout.
   * \param  p The other particle to set this one from.
   * \tparam L The layout of the other particle.
   */
  template <typename L>
  ripple_host_device Particle(const Particle<T, Dims, L>& p) noexcept
  : storage_{p.storage_} {}

  /**
   * Copy assignment from a particle of the same type.
   * \param p The other particle to copy from.
   */
  ripple_host_device auto operator=(const Particle& p) noexcept {
    storage_ = p.storage_;
  }

  /**
   * Copy assignment for particle of a different layout.
   * \param  p The other particle to set this one from.
   * \tparam L The layout of the other particle.
   */
  template <typename L>
  ripple_host_device auto operator=(const Particle<T, Dims, L>& s) noexcept {
    storage_ = s.storage_;
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets the position in the given dimension.
   * \param i The dimension to get the position for.
   */
  ripple_host_device auto x(size_t i) noexcept -> T& {
    return storage_.template get<0>(i);
  }

  /**
   * Gets the position in the given dimension.
   * \param i The dimension to get the position for.
   */
  ripple_host_device auto x(size_t i) const noexcept -> const T& {
    return storage_.template get<0>(i);
  }

  /**
   * Gets the position in the given dimension.
   * \tparam I The dimension to get the position in.
   */
  template <size_t I>
  ripple_host_device auto x() noexcept -> T& {
    return storage_.template get<0, I>();
  }

  /**
   * Gets the position in the given dimension.
   * \tparam I The dimension to get the position in.
   */
  template <size_t I>
  ripple_host_device auto x() const noexcept -> const T& {
    return storage_.template get<0, I>();
  }

  /**
   * Gets the velocity in the given dimension.
   * \param i The index of the velocity component to get.
   */
  ripple_host_device auto v(size_t i) noexcept -> T& {
    return storage_.template get<1>(i);
  }

  /**
   * Gets the velocity in the given dimension.
   * \param i The index of the velocity component to get.
   */
  ripple_host_device auto v(size_t i) const noexcept -> const T& {
    return storage_.template get<1>(i);
  }

  /**
   * Gets the velocity in the given dimension.
   * \tparam I The dimension to get the velocity component in.
   */
  template <size_t I>
  ripple_host_device auto v() noexcept -> T& {
    return storage_.template get<1, I>();
  }

  /**
   * Gets the velocity in the given dimension.
   * \tparam I The dimension to get the velocity component in.
   */
  template <size_t I>
  ripple_host_device auto v() const noexcept -> const T& {
    return storage_.template get<1, I>();
  }

  /**
   * Updates the position of the particle using the velocity and the time
   * delta.
   * \param dt The time delta to use to update the position.
   */
  ripple_host_device auto update(T dt) noexcept -> void {
    ripple::unrolled_for<dims>([this, dt](auto dim) {
      this->template x<dim>() += dt * this->template v<dim>();
    });
  }

 private:
  Storage storage_; //!< Storage of particle data.
};

#endif // RIPPLE_BENCHMARK_PARTICLE_HPP

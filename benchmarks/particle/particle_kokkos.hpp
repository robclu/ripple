#include <Kokkos_Core.hpp>
#include <ripple/utility/portability.hpp>
#include <mpi.h>

using Real = double;

// This is a test class for a particle with a number of dimensions.
template <typename T, size_t Dims>
struct Particle {
  /** Dimensions for the particle. */
  static constexpr size_t dims = Dims;

  ripple_all Particle() {}

  /**
   * Constructor from another particle.
   * \param p The other particle to set this one from.
   */
  ripple_all Particle(const Particle& p) noexcept : data_{p.data_} {}

  /**
   * Copy assignment from a particle of the same type.
   * \param p The other particle to copy from.
   */
  ripple_all auto operator=(const Particle& p) noexcept {
    data_ = p.data_;
  }

  /*==--- [interface] ------------------------------------------------------==*/

  /**
   * Gets the position in the given dimension.
   * \param i The dimension to get the position for.
   */
  ripple_all auto x(size_t i) noexcept -> T& {
    return data_[i];
  }

  /**
   * Gets the position in the given dimension.
   * \param i The dimension to get the position for.
   */
  ripple_all auto x(size_t i) const noexcept -> const T& {
    return data_[i];
  }

  /**
   * Gets the position in the given dimension.
   * \tparam I The dimension to get the position in.
   */
  template <size_t I>
  ripple_all auto x() noexcept -> T& {
    return data_[I];
  }

  /**
   * Gets the position in the given dimension.
   * \tparam I The dimension to get the position in.
   */
  template <size_t I>
  ripple_all auto x() const noexcept -> const T& {
    return data_[I];
  }

  /**
   * Gets the velocity in the given dimension.
   * \param i The index of the velocity component to get.
   */
  ripple_all auto v(size_t i) noexcept -> T& {
    return data_[dims + i];
  }

  /**
   * Gets the velocity in the given dimension.
   * \param i The index of the velocity component to get.
   */
  ripple_all auto v(size_t i) const noexcept -> const T& {
    return data_[dims + i];
  }

  /**
   * Gets the velocity in the given dimension.
   * \tparam I The dimension to get the velocity component in.
   */
  template <size_t I>
  ripple_all auto v() noexcept -> T& {
    return data_[dims + I];
  }

  /**
   * Gets the velocity in the given dimension.
   * \tparam I The dimension to get the velocity component in.
   */
  template <size_t I>
  ripple_all auto v() const noexcept -> const T& {
    return data_[dims + I];
  }

  /**
   * Updates the position of the particle using the velocity and the time
   * delta.
   * \param dt The time delta to use to update the position.
   */
  ripple_all auto update(T dt) noexcept -> void {
    for (size_t i = 0; i < dims; ++i) {
      x(i) += dt * v(i);
    }
  }

 private:
  T data_[dims * 2];
};

static constexpr size_t dimensions = 3;
using ParticleType                 = Particle<Real, dimensions>;
using ViewType                     = Kokkos::View<ParticleType*>;

using Real = double;

struct InitView {
  ViewType particles;
  int      start_index;

  InitView(ViewType p_, int i_) : particles(p_), start_index(i_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    for (size_t d = 0; d < dimensions; ++d) {
      particles(i + start_index).x(d) = i;
      particles(i + start_index).v(d) = 1.4f * d;
    }
  }
};

struct UpdateParticles {
  ViewType particles;
  Real     dt;
  int      start_index;

  UpdateParticles(ViewType p_, Real dt_, int i_)
  : particles(p_), dt(dt_), start_index(i_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    particles(i + start_index).update(dt);
  }
};
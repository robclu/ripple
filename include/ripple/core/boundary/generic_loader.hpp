//==--- ripple/core/boundary/generic_loader.hpp ------------ -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  generic_loader.hpp
/// \brief This file defines an implementation of a boundary loader which
///        is generic and can be configured.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BOUNDARY_GENERIC_LOADER_HPP
#define RIPPLE_BOUNDARY_GENERIC_LOADER_HPP

#include "boundary_loader.hpp"

namespace ripple {

/** Defines the location of the boundary. */
enum class BoundaryLocation : uint8_t {
  front = 0, //!< Front of the domain.
  back  = 1  //!< Back of the domain.
};

/** Defines the type of the boundary. */
enum class BoundaryType : uint8_t {
  transmissive = 0x00,
  reflective   = 0x01,
  zero         = 0x02
};

/**
 * The GenericLoader is an implementation of an BoundaryLoader which can be
 * configured to have different  behaviour at different boundary locations.
 */
class GenericLoader : public BoundaryLoader<GenericLoader> {
  uint64_t config = 0; // Configuration for the loader.

  // clang-format off
  /** Number of bits in each type of boundary. */
  static constexpr size_t bits_per_type = 8;
  /** Front and back for each type. */
  static constexpr size_t bits_per_dim  = bits_per_type * 2;
  // clang-format on

  /**
   * Gets the type of the boundary.
   * \param  dim      The dimension to get the type for.
   * \param  location The location of the boundary in the dimension.
   * \tparam Dim      The type of the dimension specifier.
   * \return The type of the boundary.
   */
  template <typename Dim>
  ripple_host_device auto constexpr get_boundary_type(
    Dim&& dim, BoundaryLocation location) const noexcept -> BoundaryType {
    const auto shift_amount = static_cast<size_t>(dim) * bits_per_dim +
                              static_cast<size_t>(location) * bits_per_type;

    return static_cast<BoundaryType>((config >> shift_amount) & 0xFF);
  }

 public:
  /**
   * Contigures the loader for the  given dimension to have the given location
   * and type for the dimension.
   * \param  dim      The dimension to confiure.
   * \param  location The location to configure.
   * \param  type     The type of the boundary at the location.
   * \tparam Dim      The type of the dimension specifier.
   * \return A refernce to the loader.
   */
  template <typename Dim>
  constexpr auto
  configure(Dim&& dim, BoundaryLocation location, BoundaryType type) noexcept
    -> GenericLoader& {
    const auto shift_amount = static_cast<size_t>(dim) * bits_per_dim +
                              static_cast<size_t>(location) * bits_per_type;

    config |= static_cast<uint8_t>(type) << shift_amount;
    return *this;
  }

  /**
   * Gets the configuration value of the loader, useful for debugging.
   * \return The configuration of the loader.
   */
  constexpr auto configuration() const noexcept -> uint64_t {
    return config;
  }

  /**
   * Loads the front boundary in the given dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the boundary cell to load.
   * \param  index    The index of the boundary cell in the dimension.
   * \param  dim      The dimension to load the boundary in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto
  load_front(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    static_assert_iterator(it);
    *it = *it.offset(ripple_forward(dim), index);
    switch (get_boundary_type(dim, BoundaryLocation::front)) {
      case BoundaryType::reflective: {
        it->set_v(ripple_forward(dim), -1 * it->v(ripple_forward(dim)));
        return;
      }
      case BoundaryType::zero: {
        it->set_v(ripple_forward(dim), 0);
        return;
      }
      default: return;
    }
  }

  /**
   * Loads the back boundary in the given dimension, using the value of
   * the index in the dimension to find the appropriate cell.
   * \param  it       An iterator to the boundary cell to load.
   * \param  index    The index of the boundary cell in the dimension.
   * \param  dim      The dimension to load the boundary in.
   * \tparam Iterator The type of the iterator.
   * \tparam Dim      The type of the dimension specifier.
   */
  template <typename Iterator, typename Dim>
  ripple_host_device constexpr auto
  load_back(Iterator&& it, int index, Dim&& dim) const noexcept -> void {
    static_assert_iterator(it);
    *it = *it.offset(ripple_forward(dim), index);
    switch (get_boundary_type(dim, BoundaryLocation::back)) {
      case BoundaryType::reflective: {
        it->set_v(ripple_forward(dim), -1 * it->v(ripple_forward(dim)));
        return;
      }
      case BoundaryType::zero: {
        it->set_v(ripple_forward(dim), 0);
        return;
      }
      default: return;
    }
  }
};

} // namespace ripple

#endif // RIPPLE_BOUNDARY_GENERIC_LOADER_HPP
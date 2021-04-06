/**=--- ripple/graph/stealer.hpp --------------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  stealer.hpp
 * \brief This file implements functionality for determining which thread to
 *        steal from.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_GRAPH_STEALER_HPP
#define RIPPLE_GRAPH_STEALER_HPP

#include <ripple/math/math.hpp>

namespace ripple {

/**
 * Defines the type of available work stealing policies.
 */
enum class StealPolicy {
  random      = 0, //!< Randomly steal from a thread.
  neighbour   = 1, //!< Steal from thread's neighbour.
  topological = 2  //!< Steal based on topology.
};

/**
 * Struct which can determine which thread to steal from given a thread index
 * and the number of threads available.
 * \tparam Policy The stealing policy.
 */
template <StealPolicy Policy>
struct Stealer {};

/*==--- [specializations] --------------------------------------------------==*/

/**
 * Specialization for a random stealing policy.
 */
template <>
struct Stealer<StealPolicy::random> {
  /**
   * Overload of call operator to get the index to steal from.
   * \param curr_id   The current index.
   * \param total_ids The total number of all indices.
   * \return The random index of the thread to steal from.
   */
  auto
  operator()(uint32_t curr_id, uint32_t total_ids) const noexcept -> uint32_t {
    return math::randint(uint32_t{0}, total_ids - 1);
  }
};

/**
 * Specialization for a nearest neighbour stealing policy. This steal from the
 * next thread index, wrapped around.
 */
template <>
struct Stealer<StealPolicy::neighbour> {
  /**
   * Overload of call operator to get the index to steal from.
   * \param curr_id   The current index.
   * \param total_ids The total number of all indices.
   * \return The index of the neighbour ro steal from.
   */
  auto
  operator()(uint32_t curr_id, uint32_t total_ids) const noexcept -> uint32_t {
    return (curr_id + uint32_t{1}) % total_ids;
  }
};

/**
 * Specialization for a topological stealing policy.
 */
template <>
struct Stealer<StealPolicy::topological> {
  /**
   * Overload of call operator to return the index to steal from.
   *
   * \note Currently this just does nearest neighbour stealing.
   * \todo Implement topological stealing.
   *
   * \param curr_id   The current index.
   * \param total_ids The total number of all indices.
   * \return The index of the thread to steal from.
   */
  auto
  operator()(uint32_t curr_id, uint32_t total_ids) const noexcept -> uint32_t {
    return (curr_id + uint32_t{1}) % total_ids;
  }
};

/*==--- [aliases] ----------------------------------------------------------==*/

// clang-format off
/** Alias for a randon stealer. */
using RandomStealer    = Stealer<StealPolicy::random>;
/** Alias for a nearest neighbour stealer. */
using NeighbourStealer = Stealer<StealPolicy::neighbour>;
/** Alias for a topological stealer. */
using TopoStealer      = Stealer<StealPolicy::topological>;
// clang-format on

} // namespace ripple

#endif // RIPPLE_GRAPH_STEALER_HPP

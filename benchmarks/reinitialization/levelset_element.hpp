//==--- ripple/benchmarks/levelset_element.hpp ------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_element.hpp
/// \brief This file defines a levelset element class for benchmarks.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_BENCHMARK_LEVELSET_ELEMENT_HPP
#define RIPPLE_BENCHMARK_LEVELSET_ELEMENT_HPP

#include <ripple/core/storage/storage_descriptor.hpp>
#include <ripple/core/storage/storage_traits.hpp>
#include <ripple/core/storage/struct_accessor.hpp>
#include <ripple/core/utility/portability.hpp>

/**
 * Defines possible states for levelset reinitialization.
 */
enum class State : uint32_t { source = 0, converged = 1, updatable = 2 };

/**
 * Defines an element for a levelset, which has a value and a state for whether
 * the element is converged.
 * \param T The type of the data for the levelset element.
 */
template <typename T, typename Layout = ripple::StridedView>
struct LevelsetElement
: public ripple::PolymorphicLayout<LevelsetElement<T, Layout>> {
  /**
   * Defines the types to store for FIM. We need the actual type, and a boolean
   * to represent if the cell has converged.
   */
  using Descriptor = ripple::StorageDescriptor<Layout, T, State>;
  /** Type of the storage for the element. */
  using Storage = typename Descriptor::Storage;

  Storage storage; //!< Storage for the element.

  /**
   * Constructor to set the element from another element.
   * \param e The other element to set from.
   */
  ripple_all LevelsetElement(const LevelsetElement& e) noexcept
  : storage(e.storage) {}

  /**
   * Constructor to set the element from the storage.
   * \param s The storage to set the element from.
   */
  ripple_all LevelsetElement(const Storage& s) noexcept : storage(s) {}

  /**
   * Overload of assignment operator which copies the storage.
   * \param other The other element to set this one from.
   */
  ripple_all auto
  operator=(const LevelsetElement& other) noexcept -> LevelsetElement& {
    storage = other.storage;
    return *this;
  }

  /**
   * Overload of assignment operato which sets the value for the element to the
   * given value.
   * \param value The value to set the element to.
   */
  ripple_all auto operator=(T val) noexcept -> LevelsetElement& {
    value() = val;
    return *this;
  }

  /**
   * Returns the value of the levelset element.
   */
  ripple_all auto value() noexcept -> T& {
    return storage.template get<0>();
  }

  /**
   * Returns the value of the element.
   */
  ripple_all auto value() const noexcept -> const T& {
    return storage.template get<0>();
  }

  /**
   * Returns the state of the element.
   */
  ripple_all auto state() -> State& {
    return storage.template get<1>();
  }

  /**
   * Returns the state of the element.
   */
  ripple_all auto state() const -> const State& {
    return storage.template get<1>();
  }
};

#endif // RIPPLE_BENCHMARK_LEVELSET_ELEMENT_HPP
